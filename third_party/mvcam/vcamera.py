import logging
import os
import sys
import time
from typing import Union, List, Callable

import cv2
import numpy as np
import tqdm

import mvsdk


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class vCamera:
    _device_info: mvsdk.tSdkCameraDevInfo
    _cap: Union[None, mvsdk.tSdkCameraCapbility]
    p_frame_buffer: int = 0
    h_camera: int = 0
    width: int = 0
    height: int = 0
    num_channels: int = 3
    timeout_ms: int = 2000

    def __init__(self,
                 device_info: mvsdk.tSdkCameraDevInfo,
                 exposure_ms: float = 30,
                 trigger: int = 0,
                 resolution_preset: int = 0,
                 frame_speed: int = 0,
                 transform: Callable = None) -> None:
        self._device_info = device_info
        self._cap = None

        self.trigger = trigger  # 0 auto, 1 soft, 2 hard
        self.auto_exposure = 1 if exposure_ms <= 0 else 0
        self.exposure_ms = exposure_ms
        self.resolution_preset = resolution_preset
        self.frame_speed = frame_speed
        self.transform = transform

        self.h_camera = 0
        self.p_frame_buffer = 0

        self._opened: bool = False

    @property
    def device_info(self):
        return self._device_info

    @property
    def capacity(self):
        return self._cap

    @property
    def SN(self):
        return str(self.device_info.acSn, encoding='ascii')

    @property
    def is_mono(self) -> Union[None, bool]:
        if self._cap is not None:
            return self.capacity.sIspCapacity.bMonoSensor != 0
        else:
            return None

    @property
    def is_open(self) -> bool:
        return self._opened

    def open(self) -> List[int]:

        errors = []

        # Stage I: Init camera
        try:
            self.h_camera = mvsdk.CameraInit(self._device_info, -1, -1)
        except mvsdk.CameraException as e:
            logging.warning(f"CameraInit Failed({e.error_code}), {e.message}")
            return errors
        self._cap = mvsdk.CameraGetCapability(self.h_camera)
        logging.debug(f"[Recorder] SN={self.SN}, handle={self.h_camera}")

        # Stage II: Allocate buffer, according to maximum resolution
        if self.is_mono:
            errors.append(mvsdk.CameraSetIspOutFormat(self.h_camera, mvsdk.CAMERA_MEDIA_TYPE_MONO8))
        frame_buffer_size = self.capacity.sResolutionRange.iWidthMax * self.capacity.sResolutionRange.iHeightMax * (
            1 if self.is_mono else 3)
        self.p_frame_buffer = mvsdk.CameraAlignMalloc(frame_buffer_size, 16)
        self.num_channels = 1 if self.is_mono else 3

        # Stage III: Set resolution
        resolution: mvsdk.tSdkImageResolution = self.capacity.pImageSizeDesc[self.resolution_preset]
        errors.append(mvsdk.CameraSetImageResolution(self.h_camera, resolution))
        self.width = resolution.iWidth
        self.height = resolution.iHeight

        errors.append(mvsdk.CameraSetTriggerMode(self.h_camera, self.trigger))  # 相机模式连续采集
        if True:  # conti or soft
            errors.append(mvsdk.CameraSetAeState(self.h_camera,
                                                 self.auto_exposure))  # manual exposure -> 0 AUTO EXPOSURE -> 1，曝光时间
            if not self.auto_exposure:
                errors.append(
                    mvsdk.CameraSetFrameSpeed(self.h_camera, self.frame_speed))  # 长曝光时需要设置frame_speed为0（慢速）
                errors.append(mvsdk.CameraSetExposureTime(self.h_camera, int(self.exposure_ms * 1000)))
            mvsdk.CameraSetWbMode(self.h_camera, 0)

        errors.append(mvsdk.CameraPlay(self.h_camera))  # 让SDK内部取图线程开始工作

        self._opened = True
        logging.info(f"[Recorder] SN={self.SN}: opened>")
        return errors

    def close(self) -> List[int]:
        errors = []
        if self.h_camera > 0:
            errors.append(mvsdk.CameraStop(self.h_camera))
            errors.append(mvsdk.CameraUnInit(self.h_camera))
            self.h_camera = 0

        if self.p_frame_buffer != 0:
            errors.append(mvsdk.CameraAlignFree(self.p_frame_buffer))
            self.p_frame_buffer = 0

        self._opened = False
        self.p_frame_buffer = 0

        return errors

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, trace):
        if type is not None:
            logging.warning(type)
            logging.warning(value)
            logging.warning(trace)
        self.close()

    def read(self) -> Union[np.ndarray, None]:
        try:
            p_raw_data, frame_head = mvsdk.CameraGetImageBuffer(self.h_camera, self.timeout_ms)
        except Exception as e:
            return None

        mvsdk.CameraImageProcess(self.h_camera, p_raw_data, self.p_frame_buffer, frame_head)
        mvsdk.CameraReleaseImageBuffer(self.h_camera, p_raw_data)
        # 此时图片已经存储在self.p_frame_buffer中

        frame_data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(self.p_frame_buffer)
        frame: np.ndarray = np.frombuffer(frame_data, dtype=np.uint8)
        num_channels = 1 if frame_head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
        frame = frame.reshape((frame_head.iHeight, frame_head.iWidth, num_channels))
        if self.transform is not None:
            frame = self.transform(frame)
        return frame

    def read_buffer(self):
        try:
            p_raw_data, frame_head = mvsdk.CameraGetImageBuffer(self.h_camera, self.timeout_ms)
        except Exception as e:
            return None

        mvsdk.CameraImageProcess(self.h_camera, p_raw_data, self.p_frame_buffer, frame_head)
        mvsdk.CameraReleaseImageBuffer(self.h_camera, p_raw_data)
        # 此时图片已经存储在self.p_frame_buffer中

        frame_data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(self.p_frame_buffer)
        return frame_data

    @property
    def intrinsic_matrix(self):
        auto_exposure_old = self.auto_exposure
        is_opend_old = self.is_open

        self.auto_exposure = True
        if self.is_open:  # 重新开启
            self.close()
            time.sleep(0.1)

        self.open()
        chess_board_image: np.ndarray = self.read()
        logging.info("Intrinsic matrix calculated")
        self.close()

        # 还原参数
        self.auto_exposure = auto_exposure_old
        if is_opend_old:
            self.open()
        return None


class vCameraSystem:
    def __init__(self,
                 exposure_ms: int = 40,
                 trigger: int = 0,
                 resolution_preset: int = 0,
                 frame_speed: int = 0,
                 transform: Callable = None) -> None:
        self.recorder_processes = []
        self.exposure_ms = exposure_ms
        self.trigger = trigger
        self.resolution_preset = resolution_preset
        self.frame_speed = frame_speed
        self.transform = transform

    @property
    def camera_list(self):
        return mvsdk.CameraEnumerateDevice()

    def __len__(self):
        return len(mvsdk.CameraEnumerateDevice())

    def get_intrinsics(self):
        raise NotImplemented

    def __getitem__(self, key):
        return vCamera(self.camera_list[key],
                       self.exposure_ms,
                       self.trigger,
                       self.resolution_preset,
                       self.frame_speed,
                       self.transform)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # Arguments
    DevList = mvsdk.CameraEnumerateDevice()
    if len(sys.argv) < 2:
        nDev = len(DevList)
        if nDev < 1:
            logging.error("No camera was found!")
            exit()
        for i, DevInfo in enumerate(DevList):
            print(f"{i}: {DevInfo.GetFriendlyName()} {DevInfo.GetPortType()}")
        IDX = 0 if nDev == 1 else int(input("Select camera: "))
    else:
        IDX = int(sys.argv[1])

    EXPOSURE_MS = 30
    FRAME_SPEED = 0
    TRIGGER = 0
    RESOLUTION_PRESET = 0
    # SAVE_PATH_BASE = '/home/xuehan/Desktop/PhoXiCameraCPP/ExternalCamera/Data'
    SAVE_PATH_BASE = '/home/xuehan/Desktop/CoRL_vis'
    os.makedirs(SAVE_PATH_BASE, exist_ok=True)

    cam_sys = vCameraSystem(exposure_ms=EXPOSURE_MS,
                            trigger=TRIGGER,
                            resolution_preset=RESOLUTION_PRESET,
                            frame_speed=FRAME_SPEED)

    camera = cam_sys[IDX]
    with camera as c:
        with tqdm.tqdm(range(1)) as pbar:
            start_t = time.time()
            cnt = 0
            capture_cnt = 0
            cmd = cv2.waitKey(1)
            while (cmd & 0xFF) != ord('q'):
                cnt += 1
                img = c.read()
                img = img[:, :, ::-1]  # RGB -> BGR
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to grayscale
                if (cmd & 0xFF) == ord('\r'):                    
                    fname = os.path.join(SAVE_PATH_BASE, 'test{}.png'.format(capture_cnt))
                    cv2.imwrite(fname, img)
                    print('Writing to {}!'.format(fname))
                    capture_cnt += 1
                cv2.imshow("Press q to end; Press Enter to capture", img)
                pbar.set_description(f"fps={cnt / (time.time() - start_t)}")
                cmd = cv2.waitKey(1)

