SHELL = /bin/bash

MODEL_CKPT_PATH_LONG_STAGE1 = /home/xuehan/UniFolding/outputs/model_stage1_tshirt_long
MODEL_CKPT_PATH_LONG_STAGE2 = /home/xuehan/UniFolding/outputs/model_stage2_tshirt_long
MODEL_CKPT_PATH_LONG_STAGE3 = /home/xuehan/UniFolding/outputs/model_stage3_tshirt_long

MODEL_CKPT_PATH_SHORT_STAGE1 = /home/xuehan/UniFolding/outputs/model_stage1_tshirt_short
MODEL_CKPT_PATH_SHORT_STAGE2 = /home/xuehan/UniFolding/outputs/model_stage2_tshirt_short
MODEL_CKPT_PATH_SHORT_STAGE3 = /home/xuehan/UniFolding/outputs/model_stage3_tshirt_short

CONFIG_MVCAM_DEV = $(shell lsusb | grep MindVision | awk 'END { if (NR==0 || $$2=="") print "--"; else print "/dev/bus/usb/"$$2"/"$$4;}' | head -c -2)  #/dev/bus/usb/002/002
ROBOT_LEFT_CLIENT_PATH = /home/xuehan/FlexivElements_v2_10_5_left
ROBOT_RIGHT_CLIENT_PATH = /home/xuehan/FlexivElements_v2_10_5_right

.PHONY: all
all:
	@echo "Please specify a target"

.PHONY: configure-env
configure.env:
	@bash setup-env.sh

calibration.handeye:
	python tools/handeye_cali.py

calibration.external: manipulation.prerun
	python tools/external_cam_cali.py

calibration.generate_json:
	python tools/find_world_transform_from_robot_cali.py


manipulation.prerun:
	@perm=$$(ls -ld ${CONFIG_MVCAM_DEV} | cut -b 1-10); \
	if [ $$perm != "crwxrwxrwx" ]; then \
	echo "Changing permissions of ${CONFIG_MVCAM_DEV} to 777"; \
	sudo chmod 777 ${CONFIG_MVCAM_DEV}; \
	fi
	umask 002

# only training
stage1.virtual.tshirt_long.train:
	python train_virtual.py --config-name train_virtual_tshirt_long.yaml \
		hydra.job.chdir=True

# only training
stage1.virtual.tshirt_short.train:
	python train_virtual.py --config-name train_virtual_tshirt_short.yaml \
		hydra.job.chdir=True

# data collection + training
stage2.virtual.tshirt_long.run:
	python run_virtual.py \
		--config-name experiment_virtual_tshirt_long.yaml \
		hydra.job.chdir=True \
		inference.model_path=${MODEL_CKPT_PATH_LONG_STAGE1}

# data collection + training
stage2.virtual.tshirt_short.run:
	python run_virtual.py --config-name experiment_virtual_tshirt_short.yaml \
		hydra.job.chdir=True \
		inference.model_path=${MODEL_CKPT_PATH_SHORT_STAGE1}

# data collection + training
stage3.real.tshirt_long.run: manipulation.prerun
	python \
        run_real.py --config-name experiment_real_tshirt_long.yaml \
        hydra.job.chdir=True \
        experiment.strategy.random_exploration.enable=True \
        inference.model_path=${MODEL_CKPT_PATH_LONG_STAGE2}

# data collection + training
stage3.real.tshirt_short.run: manipulation.prerun
	python \
        run_real.py --config-name experiment_real_tshirt_short.yaml \
        hydra.job.chdir=True \
        experiment.strategy.random_exploration.enable=True \
        inference.model_path=${MODEL_CKPT_PATH_SHORT_STAGE2}

# for inference
stage3.real.tshirt_long.test: manipulation.prerun
	python \
        run_real.py --config-name experiment_real_tshirt_long.yaml \
        hydra.job.chdir=False \
        experiment.strategy.random_exploration.enable=False \
        experiment.strategy.trial_num_per_instance=10 \
        logging.tag=tshirt_long_test \
        inference.model_path=${MODEL_CKPT_PATH_LONG_STAGE3}

# for inference
stage3.real.tshirt_short.test: manipulation.prerun
	python \
        run_real.py --config-name experiment_real_tshirt_short.yaml \
        hydra.job.chdir=False \
        experiment.strategy.random_exploration.enable=False \
        experiment.strategy.trial_num_per_instance=10 \
        logging.tag=tshirt_short_test \
        inference.model_path=${MODEL_CKPT_PATH_SHORT_STAGE3}


# only data collection (no training)
stage3.virtual.tshirt_long.run:
	python run_virtual.py \
		--config-path config/virtual_experiment_stage3 \
		--config-name experiment_virtual_tshirt_long.yaml \
		hydra.job.chdir=True \
		inference.model_path=${MODEL_CKPT_PATH_LONG_STAGE2}


# only data collection (no training)
stage3.virtual.tshirt_short.run:
	python run_virtual.py \
		--config-path config/virtual_experiment_stage3 \
		--config-name experiment_virtual_tshirt_short.yaml \
		hydra.job.chdir=True \
		inference.model_path=${MODEL_CKPT_PATH_SHORT_STAGE2}

# for inference of ClothFunnels
clothfunnels.real.tshirt_long.test: manipulation.prerun
	python run_clothfunnels.py hydra.job.chdir=False

# for capturing garment image in canonical pose
tools.capture_canonical_tshirt_long: manipulation.prerun
	python \
		tools/capture_canonical_garment.py --config-name experiment_real_tshirt_long.yaml \
		hydra.job.chdir=False \
        experiment.runtime_training_config_override.logger.experiment_name=canonical_garment \
        logging.namespace=captures \
        logging.tag=tshirt_long_canonical_corl

# for capturing garment image in canonical pose
tools.capture_canonical_tshirt_short: manipulation.prerun
	python \
		tools/capture_canonical_garment.py --config-name experiment_real_tshirt_short.yaml \
		hydra.job.chdir=False \
        experiment.runtime_training_config_override.logger.experiment_name=canonical_garment \
        logging.namespace=captures \
        logging.tag=tshirt_short_canonical_corl

robot.console.right:
	@cd ${ROBOT_RIGHT_CLIENT_PATH} && bash run_FlexivElements.sh

robot.console.left:
	@cd ${ROBOT_LEFT_CLIENT_PATH} && bash run_FlexivElements.sh

debug.controller:
	python tools/debug_controller.py

.PHONY: clean
clean:
	@echo "Cleaning up..."
