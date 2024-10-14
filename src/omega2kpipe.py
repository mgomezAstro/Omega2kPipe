import numpy as np
from datetime import datetime
from astropy.io import fits
from astropy.stats import mad_std, SigmaClip
from astropy.table import unique
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u
import ccdproc
from astropy.nddata import CCDData
from pathlib import Path
import os
import logging
import warnings


warnings.filterwarnings(action="ignore", message="RuntimeWarning")
warnings.simplefilter("ignore", category=AstropyWarning)


def remove_weird_keywords(list_of_fits: list):
    weird_keywords = ["PRESS1", "PRESS2"]
    for f in list_of_fits:
        hdu = fits.open(f)
        for k in weird_keywords:
            if k in hdu[0].header:
                del hdu[0].header[k]
        hdu.writeto(f, overwrite=True)


class Orginizer:
    def __init__(self, raw_dir: str = None):
        self.raw_dir = Path.cwd()

        self._mkdir_dark()
        self._mkdir_flat()
        self._mkdir_science()
        self._mkdir_sky()
        self._mkdir_mask()

        if raw_dir is not None:
            self.raw_dir / raw_dir

    def _mkdir_dark(self):
        if not os.path.exists(self.raw_dir / "dark"):
            os.mkdir(self.raw_dir / "dark")
        self.dark_dir = self.raw_dir / "dark"

    def _mkdir_flat(self):
        if not os.path.exists(self.raw_dir / "flat"):
            os.mkdir(self.raw_dir / "flat")
        self.flat_dir = self.raw_dir / "flat"

    def _mkdir_sky(self):
        self.sky_dir = self.raw_dir / "sky"
        if not os.path.exists(self.sky_dir):
            os.mkdir(self.sky_dir)

    def _mkdir_science(self):
        self.sci_dir = self.raw_dir / "sci"
        if not os.path.exists(self.sci_dir):
            os.mkdir(self.sci_dir)

    def _mkdir_mask(self):
        self.mask_dir = self.raw_dir / "mask"
        if not os.path.exists(self.mask_dir):
            os.mkdir(self.mask_dir)


class Omega2kPipe(Orginizer):
    def __init__(self, log_level: int = 2):
        super().__init__()

        self.cali_fits = ccdproc.ImageFileCollection(
            location=self.raw_dir, glob_include="*cal-*.fits"
        )
        self.sci_fits = ccdproc.ImageFileCollection(
            location=self.raw_dir, glob_include="*sci-*.fits"
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level * 10)
        ch = logging.StreamHandler()
        ch.setLevel(log_level * 10)
        formatter = logging.Formatter(
            "%(levelname)s | %(asctime)s | %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    @property
    def _weird_keywords(self):
        return ["PRESS1", "PRESS2"]

    def _history_keywords(
        self,
        hdu,
        name: str,
        list_of_calibs: list,
        imtype: str,
        band: str = None,
    ):
        hdu.header["FILENAME"] = f"{name}.fits"
        hdu.header["OBJECT"] = name
        hdu.header["HISTORY"] = f"--- {imtype} ---"
        if band is not None:
            hdu.header["HISTORY"] = f"FILTER {band}"
        hdu.header["HISTORY"] = (
            f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        hdu.header["HISTORY"] = (
            f"{imtype} created with {len(list_of_calibs)} images."
        )
        hdu.header["HISTORY"] = (
            "Created using the ccdproc.combine average and sigma clipping."
        )
        for calib in list_of_calibs:
            hdu.header["HISTORY"] = f"    {calib.split('/')[-1]}"

        return hdu

    def _clean_bad_pixels(self, data, mask):
        rows, cols = mask.shape

        image = data.data
        index_error = 0
        for i in range(rows):
            for j in range(cols):
                if mask[i, j] == 1:
                    try:
                        image[i, j] = np.nan
                        box_im = image[i - 5 : i + 5, j - 5 : j + 5]
                        mean_value = np.nanmedian(box_im)
                        image[i, j] = mean_value
                    except IndexError:
                        index_error += 1
        if index_error > 0:
            self.logger.warning(
                f"A total of {index_error} pixels at the edge of the image were not correctly masked."
            )

        data.data = image

        return data

    def get_master_dark(self):
        mask_darks = self.cali_fits.summary["filter"] == "BLANK"
        dark_itime_exptime = self.cali_fits.summary["itime", "exptime"][
            mask_darks
        ]
        dark_itime_exptime = unique(
            dark_itime_exptime, keys=["itime", "exptime"]
        )

        for itime, exptime in dark_itime_exptime:
            dark_list = self.cali_fits.files_filtered(
                filter="BLANK", itime=itime, exptime=exptime, include_path=True
            )

            master_dark = ccdproc.combine(
                dark_list,
                method="average",
                sigma_clip=True,
                sigma_clip_low_thresh=5.0,
                sigma_clip_high_thresh=5.0,
                sigma_clip_func=np.ma.median,
                sigma_clip_dev_func=mad_std,
                unit="adu",
            )

            output_fits_name = f"master_dark_itime_{itime}_exp_{exptime}.fits"
            master_dark.meta["combined"] = True
            master_dark = self._history_keywords(
                hdu=master_dark,
                name=output_fits_name,
                list_of_calibs=dark_list,
                imtype="DARK",
            )

            for keyword in self._weird_keywords:
                if keyword in master_dark.header:
                    del master_dark.header[keyword]

            master_dark.write(self.dark_dir / output_fits_name, overwrite=True)

        self.logger.info(
            f"Master darks successfully generated at: {self.dark_dir}."
        )

    def get_master_flats(self, dark_fits_file: str = None):
        mask_flats = self.cali_fits.summary["filter"] != "BLANK"
        flat_band = self.cali_fits.summary["filter"][mask_flats]
        flat_band = set(flat_band.tolist())

        if dark_fits_file is not None:
            master_dark = CCDData.read(dark_fits_file, unit="adu")
        else:
            darks = ccdproc.ImageFileCollection(
                location=self.dark_dir, glob_include="master*.fits"
            )
            master_dark = CCDData.read(
                darks.files_filtered(include_path=True, combined=True)[0]
            )

        flatimages = {}

        for band in flat_band:

            if band not in flatimages:
                flatimages[band] = []

            raw_flats = self.cali_fits.files_filtered(
                filter=band, include_path=True
            )

            for flat in raw_flats:
                ccd_flat = CCDData.read(flat, unit="adu")
                d_ccd_flat = ccdproc.subtract_dark(
                    ccd_flat,
                    master_dark,
                    exposure_time="exptime",
                    exposure_unit=u.second,
                    scale=True,
                )

                d_ccd_flat.header["HISTORY"] = (
                    "DARK removed, scaled dark, ccdproc."
                )
                d_ccd_flat.header["HISTORY"] = "  dark/master_dark.fits"
                d_flat_outname = "d_" + flat.split("/")[-1]
                d_ccd_flat.write(
                    self.flat_dir / d_flat_outname, overwrite=True
                )

                flatimages[band].append(d_ccd_flat)

            master_flat = ccdproc.combine(
                flatimages[band],
                method="average",
                scale=lambda x: 1 / np.nanmedian(x),
                sigma_clip=True,
                sigma_clip_high_thresh=5.0,
                sigma_clip_low_thresh=5.0,
                sigma_clip_func=np.ma.median,
                sigma_clip_dev_func=mad_std,
            )

            master_flat.meta["combined"] = True
            master_flat = self._history_keywords(
                hdu=master_flat,
                name=f"master_flat_{band}",
                list_of_calibs=raw_flats,
                imtype="FLAT",
            )

            master_flat_outname = f"master_flat_{band}.fits"
            master_flat.write(
                self.flat_dir / master_flat_outname, overwrite=True
            )

            self.logger.info(
                f"Master {band} flat generated at: {self.flat_dir} as master_flat_{band}.fits."
            )

    def reduce_images(self, clean_bp: bool = True, clean_cr: bool = True):
        darks = ccdproc.ImageFileCollection(location=self.dark_dir)
        flats = ccdproc.ImageFileCollection(
            location=self.flat_dir, glob_include="master*.fits"
        )

        for sci_path in self.sci_fits.files_filtered(include_path=True):
            sci_im = CCDData.read(sci_path, unit=u.adu)

            itime = sci_im.header["itime"]
            exptime = sci_im.header["exptime"]
            band = sci_im.header["filter"]

            dark = darks.files_filtered(
                itime=itime, exptime=exptime, include_path=True
            )
            master_dark = CCDData.read(dark[0])
            flat = flats.files_filtered(filter=band, include_path=True)
            master_flat = CCDData.read(flat[0])

            # Dark correction
            d_sci_im = ccdproc.subtract_dark(
                sci_im,
                master_dark,
                exposure_time="exptime",
                exposure_unit=u.second,
                scale=False,
            )
            d_sci_im.header["HISTORY"] = (
                "DARK removed, no scaled dark, ccdproc."
            )
            d_sci_im.header["HISTORY"] = (
                f"  dark/master_dark_itime_{itime}_exp_{exptime}.fits"
            )
            d_sci_oname = "d_" + sci_path.split("/")[-1]
            d_sci_im.write(self.sci_dir / d_sci_oname, overwrite=True)

            # Flat correction
            fd_sci_im = ccdproc.flat_correct(d_sci_im, master_flat)
            fd_sci_im.header["HISTORY"] = "FLAT corrected, ccdproc."
            fd_sci_im.header["HISTORY"] = f"  flat/master_flat_{band}.fits"
            fd_sci_im.write(
                self.sci_dir / str("f" + d_sci_oname), overwrite=True
            )

        self.logger.info(
            f"Total reduced images: {len(self.sci_fits.summary)}."
        )

    def remove_sky(self):
        all_sci_im = ccdproc.ImageFileCollection(
            location=self.sci_dir, glob_include="fd_*.fits"
        )

        bands = set(all_sci_im.summary["filter"].tolist())

        for band in bands:
            sci_im_band = all_sci_im.files_filtered(
                include_path=True, filter=band
            )

            combined_stacked_data = ccdproc.combine(
                sci_im_band,
                method="median",
                sigma_clip=True,
                scale=lambda x: 1 / np.nanmedian(x),
            )
            combined_stacked_data.write(
                self.sky_dir / f"sky_{band}.fits", overwrite=True
            )

            for sci_im in sci_im_band:
                im = CCDData.read(sci_im)
                sigclip = SigmaClip(sigma=3.0, cenfunc="median")
                median_value = np.ma.median(sigclip(im.data))
                data_skysub = (
                    im.data - combined_stacked_data.data * median_value
                )
                im.data = data_skysub
                im.header["HISTORY"] = "Sky subtracted."
                im.header["HISTORY"] = f"   sky/sky_{band}.fits"
                image_outname = "s" + sci_im.split("/")[-1]
                im.write(self.sci_dir / image_outname, overwrite=True)

        self.logger.info("Sky succesfully subtracted from all images.")
