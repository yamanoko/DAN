from torch.optim import Adam
from basic.transforms import aug_config
from OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from OCR.document_OCR.dan.trainer_dan import Manager
from OCR.document_OCR.dan.models_dan import GlobalHTADecoder
from basic.models import FCN_Encoder
from basic.scheduler import exponential_dropout_scheduler, linear_scheduler
import torch
import numpy as np
import random

def get_training_dataset():
	dataset_name = "READ_2016"  # ["RIMES", "READ_2016"]
	dataset_level = "page"  # ["page", "double_page"]
	dataset_variant = "_sem"

	# max number of lines for synthetic documents
	max_nb_lines = {
		"RIMES": 40,
		"READ_2016": 30,
	}

	dataset_params = {
				"dataset_manager": OCRDatasetManager,
				"dataset_class": OCRDataset,
				"use_ddp": False,
				"batch_size": 1,
				"valid_batch_size": 4,
				"test_batch_size": 4,
				"num_gpu": torch.cuda.device_count(),
				"worker_per_gpu": 4,
				"datasets": {
					dataset_name: "../../Datasets/formatted/{}_{}{}".format(dataset_name, dataset_level, dataset_variant),
				},
				"train": {
					"name": "{}-train".format(dataset_name),
					"datasets": [(dataset_name, "train"), ],
				},
				"valid": {
					"{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
				},
				"config": {
					"load_in_memory": True,  # Load all images in CPU memory
					"worker_per_gpu": 4,  # Num of parallel processes per gpu for data loading
					"width_divisor": 8,  # Image width will be divided by 8
					"height_divisor": 32,  # Image height will be divided by 32
					"padding_value": 0,  # Image padding value
					"padding_token": None,  # Label padding value
					"charset_mode": "seq2seq",  # add end-of-transcription ans start-of-transcription tokens to charset
					"constraints": ["add_eot", "add_sot"],  # add end-of-transcription ans start-of-transcription tokens in labels
					"normalize": False,  # Normalize with mean and variance of training dataset
					"preprocessings": [
						{
							"type": "to_RGB",
							# if grayscaled image, produce RGB one (3 channels with same value) otherwise do nothing
						},
					],
					"augmentation": aug_config(0.9, 0.1),
					# "synthetic_data": None,
					"synthetic_data": {
						"init_proba": 0.9,  # begin proba to generate synthetic document
						"end_proba": 0.2,  # end proba to generate synthetic document
						"num_steps_proba": 200000,  # linearly decrease the percent of synthetic document from 90% to 20% through 200000 samples
						"proba_scheduler_function": linear_scheduler,  # decrease proba rate linearly
						"start_scheduler_at_max_line": True,  # start decreasing proba only after curriculum reach max number of lines
						"dataset_level": dataset_level,
						"curriculum": True,  # use curriculum learning (slowly increase number of lines per synthetic samples)
						"crop_curriculum": True,  # during curriculum learning, crop images under the last text line
						"curr_start": 0,  # start curriculum at iteration
						"curr_step": 10000,  # interval to increase the number of lines for curriculum learning
						"min_nb_lines": 1,  # initial number of lines for curriculum learning
						"max_nb_lines": max_nb_lines[dataset_name],  # maximum number of lines for curriculum learning
						"padding_value": 255,
						# config for synthetic line generation
						"config": {
							"background_color_default": (255, 255, 255),
							"background_color_eps": 15,
							"text_color_default": (0, 0, 0),
							"text_color_eps": 15,
							"font_size_min": 35,
							"font_size_max": 45,
							"color_mode": "RGB",
							"padding_left_ratio_min": 0.00,
							"padding_left_ratio_max": 0.05,
							"padding_right_ratio_min": 0.02,
							"padding_right_ratio_max": 0.2,
							"padding_top_ratio_min": 0.02,
							"padding_top_ratio_max": 0.1,
							"padding_bottom_ratio_min": 0.02,
							"padding_bottom_ratio_max": 0.1,
						},
					}
				}
			}

	small_model_dataset = dataset_params["dataset_manager"](dataset_params)
	small_model_dataset.load_datasets()
	small_model_dataset.load_ddp_samplers()
	small_model_dataset.load_dataloaders()
	small_model_dataset.train_dataset.training_info = {
		"epoch": 0,
		"step": 0,
	}
	return small_model_dataset