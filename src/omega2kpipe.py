import numpy as np
from datetime import datetime
from astropy.io import fits
from astropy.stats import mad_std
import astropy.units as u
import ccdproc
from ccdproc import CCDData
from glob import glob
from pathlib import Path
import os
import shutil


class Orginizer:
    def __init__(self, raw_dir: str = None, sky_frames: list = None):
        self.raw_dir = Path.cwd()
        self.sky_frames = sky_frames

        self._mkdir_dark()
        self._mkdir_flat()
        self._mkdir_science()

        if raw_dir is not None:
            self.raw_dir / raw_dir
        if sky_frames is not None:
            self._mkdir_sky()

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

    def orgnize(self):
        all_cals = glob(str(self.raw_dir) + "/*cal-cali.fits")
        all_sci = glob(str(self.raw_dir) + "/*sci*.fits")
        for f in all_cals:
            hdr = fits.getheader(str(f))
            if hdr["FILTER"] == "BLANK":
                shutil.copy(f, self.dark_dir)
            else:
                shutil.copy(f, self.flat_dir)
        for f in all_sci:
            shutil.copy(f, self.sci_dir)


class Omega2kPipe(Orginizer):
    def __init__(self, sky_frames: list):
        super().__init__(sky_frames=sky_frames)

        self.orgnize()

        self.master_dark_path = None
        self.master_sky_path = None
        self.master_flat_path = None
        self.mask_bad_pixels_path = None

        self.master_dark = None
        self.master_sky = None
        self.master_flat = None
        self.mask_bad_pixels = None

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

    def get_master_dark(self):
        list_of_darks = glob(str(self.dark_dir) + "/*.fits")
        darkimages = []

        for dark in list_of_darks:
            data, hdr = fits.getdata(dark, header=True)
            darkimages.append(CCDData(data=data, header=hdr, unit="adu"))

        master_dark = ccdproc.combine(
            img_list=darkimages,
            method="average",
            sigma_clip=True,
            sigma_clip_low_thresh=5.0,
            sigma_clip_high_thresh=5.0,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
        )

        master_dark = self._history_keywords(
            master_dark, "master_dark", list_of_darks, "DARK"
        )

        for keyword in self._weird_keywords:
            if keyword in master_dark.header:
                del master_dark.header[keyword]

        master_dark.write(self.dark_dir / "master_dark.fits", overwrite=True)
        self.master_dark_path = self.dark_dir / "master_dark.fits"
        self.master_dark = master_dark

    def get_bad_pixels_mask(self):
        std = np.std(self.master_dark.data)

        mask = np.zeros_like(self.master_dark.data)
        mask[self.master_dark.data > 3 * std] = 1

        hdu = fits.PrimaryHDU(data=mask)
        hdu.writeto(self.sci_dir / "mask_bad_pixels.fits", overwrite=True)
        self.mask_bad_pixels_path = self.sci_dir / "mask_bad_pixels.fits"
        self.mask_bad_pixels = mask

    def get_master_flats(self):
        list_of_flats = glob(str(self.flat_dir) + "/*.fits")
        flatimages = {}

        for flat in list_of_flats:
            data, hdr = fits.getdata(flat, header=True)
            band = hdr["FILTER"]
            if band not in flatimages:
                flatimages[band] = []
            flatimages[band].append(CCDData(data=data, header=hdr, unit="adu"))

        self.master_flat = {}
        self.master_flat_path = {}
        for band in flatimages:

            for i in range(len(flatimages[band])):
                flatimages[band][i] = ccdproc.subtract_dark(
                    flatimages[band][i],
                    self.master_dark,
                    exposure_time="EXPTIME",
                    exposure_unit=u.second,
                    scale=False,
                )
                flatimages[band][i].header[
                    "HISTORY"
                ] = "Dark removed: dark/master_dark.fits"

            master_flat = ccdproc.combine(
                img_list=flatimages[band],
                method="average",
                sigma_clip=True,
                sigma_clip_low_thresh=5.0,
                sigma_clip_high_thresh=5.0,
                sigma_clip_func=np.ma.median,
                sigma_clip_dev_func=mad_std,
            )

            master_flat = master_flat.divide(
                np.median(master_flat.data, axis=0)
            )
            master_flat = self._history_keywords(
                master_flat,
                f"master_flat_{band}",
                ["flats/flat", "flats/flat"],
                "FLAT",
            )
            master_flat.write(
                self.flat_dir / f"master_flat_{band}.fits", overwrite=True
            )
            self.master_flat_path[band] = (
                self.flat_dir / f"master_flat_{band}.fits"
            )
            self.master_flat[band] = master_flat

    def reduce_images(self):
        list_of_sci = glob(str(self.sci_dir) + "/*.fits")
        sciimages = {}

        for sci in sciimages:
            data, hdr = fits.getdata(sci, header=True)
            band = hdr["FILTER"]
            if band not in sciimages:
                sciimages[band] = []

            # Processes
            sci_red = CCDData(data=data, header=hdr, unit="adu")
            sci_red = ccdproc.subtract_dark(
                sci_red,
                self.master_dark,
                exposure_time="EXPTIME",
                exposure_unit=u.seconds,
            )
            sci_red = ccdproc.flat_correct(sci_red, self.master_flat[band])

            sciimages[band].append(sci_red)
