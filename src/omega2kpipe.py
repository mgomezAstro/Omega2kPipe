import numpy as np
from datetime import datetime
from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import unique
import astropy.units as u
import ccdproc
import astroscrappy as lacos
from astropy.nddata import CCDData
from glob import glob
from pathlib import Path
import os
import shutil
import logging
import warnings


warnings.filterwarnings(action="ignore", message="RuntimeWarning")


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

    # def orgnize(self):
    # all_cals = glob(str(self.raw_dir) + "/*cal-cali.fits")
    # all_sci = glob(str(self.raw_dir) + "/*sci*.fits")
    # for f in all_cals:
    #     hdr = fits.getheader(str(f))
    #     if hdr["FILTER"] == "BLANK":
    #         filepath = self.dark_dir / f.split("/")[-1]
    #         if not filepath.is_file():
    #             shutil.copy(f, self.dark_dir)
    #     else:
    #         filepath = self.flat_dir / f.split("/")[-1]
    #         if not filepath.is_file():
    #             shutil.copy(f, self.flat_dir)
    # for f in all_sci:
    #     filepath = self.sci_dir / f.split("/")[-1]
    #     if not filepath.is_file():
    #         shutil.copy(f, self.sci_dir)

    # return ccdproc.ImageFileCollection(location=self.raw_dir)


class Omega2kPipe(Orginizer):
    def __init__(self, log_level: int = 2):
        super().__init__()

        # self.raw_fits: ccdproc.ImageFileCollection = self.orgnize()
        self.cali_fits = ccdproc.ImageFileCollection(
            location=self.raw_dir, glob_include="*cal-cali.fits"
        )
        self.sci_fits = ccdproc.ImageFileCollection(
            location=self.raw_dir, glob_include="*sci-blan.fits"
        )

        # self.master_dark_path = None
        # self.master_sky_path = None
        # self.master_flat_path = None
        # self.mask_bad_pixels_path = None

        self.master_dark = None
        self.master_sky = None
        self.master_flat = None
        self.mask_bad_pixels = None

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
                filter="BLANK", itime=itime, exptime=exptime
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

        # list_of_darks = glob(str(self.dark_dir) + "/*cali.fits")
        # darkimages = []

        # for dark in list_of_darks:
        #     data, hdr = fits.getdata(dark, header=True)
        #     darkimages.append(CCDData(data=data, header=hdr, unit="adu"))

        # master_dark = ccdproc.combine(
        #     img_list=darkimages,
        #     method="average",
        #     sigma_clip=True,
        #     sigma_clip_low_thresh=5.0,
        #     sigma_clip_high_thresh=5.0,
        #     sigma_clip_func=np.ma.median,
        #     sigma_clip_dev_func=mad_std,
        # )

        # master_dark = self._history_keywords(
        #     master_dark, "master_dark", list_of_darks, "DARK"
        # )

        # for keyword in self._weird_keywords:
        #     if keyword in master_dark.header:
        #         del master_dark.header[keyword]

        # master_dark.write(self.dark_dir / "master_dark.fits", overwrite=True)
        # self.master_dark_path = self.dark_dir / "master_dark.fits"
        # self.master_dark = master_dark

        self.logger.info(
            f"Master darks successfully generated at: {self.dark_dir}."
        )

    def get_bad_pixels_mask(self):
        std = np.nanstd(self.master_dark.data)

        mask = np.zeros_like(self.master_dark.data)
        mask[self.master_dark.data > 3 * std] = 1
        mask[self.master_dark.data < -3 * std] = 1

        hdu = fits.PrimaryHDU(data=mask)
        hdu.writeto(self.sci_dir / "mask_bad_pixels.fits", overwrite=True)
        self.mask_bad_pixels_path = self.sci_dir / "mask_bad_pixels.fits"
        self.mask_bad_pixels = mask

        self.logger.info(
            f"Bad pixel image generated and saved at: {self.mask_bad_pixels_path}. Num of bad pixels found {self.mask_bad_pixels.sum():.0f}."
        )

    def get_master_flats(self, subtract_dark: bool = True):
        mask_flats = self.cali_fits.summary["filter"] != "BLANK"
        flat_band = self.cali_fits.summary["filter"][mask_flats]
        flat_band = unique(flat_band, keys=["filter"])

        if subtract_dark:
            pass
            # list_of_flats = glob(str(self.flat_dir) + "/*cali.fits")
            # flatimages = {}
            # flatnames = {}

            # for flat in list_of_flats:
            #     data, hdr = fits.getdata(flat, header=True)
            #     band = hdr["FILTER"]
            #     if band not in flatimages:
            #         flatimages[band] = []
            #         flatnames[band] = []
            #     flatimages[band].append(CCDData(data=data, header=hdr, unit="adu"))
            #     flatnames[band].append(flat)

            # self.master_flat = {}
            # self.master_flat_path = {}
            # for band in flatimages:

            #     for i in range(len(flatimages[band])):
            #         flatimages[band][i] = ccdproc.subtract_dark(
            #             flatimages[band][i],
            #             self.master_dark,
            #             exposure_time="EXPTIME",
            #             exposure_unit=u.second,
            #             scale=False,
            #         )
            #         flatimages[band][i].header[
            #             "HISTORY"
            #         ] = "Dark removed: dark/master_dark.fits"

            #     master_flat = ccdproc.combine(
            #         img_list=flatimages[band],
            #         method="average",
            #         sigma_clip=True,
            #         sigma_clip_low_thresh=5.0,
            #         sigma_clip_high_thresh=5.0,
            #         sigma_clip_func=np.ma.median,
            #         sigma_clip_dev_func=mad_std,
            #     )

            #     master_flat = master_flat.divide(
            #         np.median(master_flat.data, axis=0)
            #     )
            #     master_flat = self._history_keywords(
            #         master_flat,
            #         f"master_flat_{band}",
            #         flatnames[band],
            #         "FLAT",
            #         band,
            #     )
            #     master_flat.header["FILTER"] = band
            #     master_flat.write(
            #         self.flat_dir / f"master_flat_{band}.fits", overwrite=True
            #     )
            #     self.master_flat_path[band] = (
            #         self.flat_dir / f"master_flat_{band}.fits"
            #     )
            #     self.master_flat[band] = master_flat

            self.logger.info(
                f"Master {band} flat generated at: {self.master_flat_path[band]}."
            )

    def reduce_images(self, clean_bp: bool = True, clean_cr: bool = True):
        list_of_sci = glob(str(self.sci_dir) + "/*sci*.fits")
        sciimages = {}

        for sci in list_of_sci:
            data, hdr = fits.getdata(sci, header=True)
            band = hdr["FILTER"]
            # objname = hdr["OBJECT"].rstrip().lstrip().replace(" ", "_")
            if band not in sciimages:
                sciimages[band] = []

            # Processes
            sci_red = CCDData(data=data, header=hdr, unit="adu")
            sci_red = ccdproc.subtract_dark(
                sci_red,
                self.master_dark,
                exposure_time="EXPTIME",
                exposure_unit=u.second,
            )
            sci_red = ccdproc.flat_correct(sci_red, self.master_flat[band])

            sci_red = self._history_keywords(
                sci_red,
                "fd_" + sci.split("/")[-1],
                [
                    self.master_dark.header["OBJECT"],
                    self.master_flat[band].header["OBJECT"],
                ],
                "SCI",
                band,
            )

            if clean_bp:
                sci_red = self._clean_bad_pixels(sci_red, self.mask_bad_pixels)

            for keyword in self._weird_keywords:
                if keyword in sci_red.header:
                    del sci_red.header[keyword]

            # if not os.path.exists(self.sci_dir / objname):
            # os.mkdir(self.sci_dir / objname)
            if not os.path.exists(self.sci_dir / "redu"):
                os.mkdir(self.sci_dir / "redu")
            outputfile_name = "fd_" + sci.split("/")[-1]
            sci_red.write(
                self.sci_dir / "redu" / outputfile_name, overwrite=True
            )

            sciimages[band].append(sci_red)

        self.logger.info(
            f"Scientific images reduced and saved at: {self.sci_dir / 'redu'}."
        )
        if clean_bp:
            self.logger.info("Bad pixels were removed.")
        self.logger.info(f"Total reduced images: {len(list_of_sci)}.")
