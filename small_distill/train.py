import os
from torch.optim import AdamW
from basic.transforms import aug_config
from OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from OCR.document_OCR.dan.trainer_dan import Manager
from OCR.document_OCR.dan.models_dan import GlobalHTADecoder
from OCR.ocr_utils import LM_ind_to_str
from basic.models import FCN_Encoder
from basic.scheduler import exponential_dropout_scheduler, linear_scheduler
from basic.utils import pad_images
from basic.metric_manager import MetricManager, keep_all_but_tokens
from basic.post_pocessing_layout import PostProcessingModuleREAD
import torch
from torch.nn import KLDivLoss
import torch.nn.functional as F
import numpy as np
import random
from transformers import VisionEncoderDecoderModel, AutoImageProcessor
from PIL import Image
from training_dataset import get_training_dataset
from teacher_dan import get_teacher_model

def resize_with_padding(image, target_height, target_width):
	"""Resizes a PIL image to the target width and height, maintaining aspect ratio and padding the remaining space.

	Args:
	image: A PIL Image object.
	target_width: The target width of the image.
	target_height: The target height of the image.

	Returns:
	A new PIL Image object with the specified dimensions and padding.
	"""

	width, height = image.size

	# Calculate the aspect ratio of the image
	aspect_ratio = width / height

	# Calculate the new dimensions based on the target size while maintaining aspect ratio
	if width / target_width > height / target_height:
		new_width = target_width
		new_height = int(target_width / aspect_ratio)
	else:
		new_height = target_height
		new_width = int(target_height * aspect_ratio)

	# Resize the image
	resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

	# Create a new image with the target dimensions and fill with padding color
	padded_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))  # White padding

	# Paste the resized image into the center of the new image
	x_offset = (target_width - new_width) // 2
	y_offset = (target_height - new_height) // 2
	padded_image.paste(resized_image, (x_offset, y_offset))

	return padded_image


if __name__ == "__main__":
	# get training dataset
	training_dataset = get_training_dataset()
	train_loader = training_dataset.train_loader
	valid_loader = training_dataset.valid_loader["READ_2016-valid"]

	# get metric manager
	dataset_name = list(training_dataset["datasets"].values())[0]
	validation_metric_manager = MetricManager(metric_names=["cer", "wer", "map_cer", "loer"], dataset_name=dataset_name)
	validation_metric_manager.post_processing_module = PostProcessingModuleREAD

	# get student model
	model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
	"microsoft/swin-tiny-patch4-window7-224",
	"distilbert/distilgpt2",
	decoder_ignore_mismatched_sizes=True,
	decoder_vocab_size=len(training_dataset.charset)+3,
	decoder_max_length=600,
	encoder_image_size=(640, 480),
	pad_token_id=training_dataset.tokens['pad'],
	eos_token_id=training_dataset.tokens['end'],
	bos_token_id=training_dataset.tokens['start'],
).to("cuda")
	image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

	# get teacher model
	dan_manager = get_teacher_model()

	# get optimizer
	optimizer = AdamW(model.parameters(), lr=1e-5)

	# define loss function
	kl_loss = KLDivLoss(reduction="batchmean")

	# hyperparameters
	num_epochs = 50000
	distill_alpha = 0.5

	best_cer = 1.0

	# training loop
	for epoch in range(num_epochs):
		# set model to training mode
		model.train()
		training_dataset.train_dataset.training_info["epoch"] = epoch
		for batch in train_loader:
			optimizer.zero_grad()
			training_dataset.train_dataset.training_info["step"] += 1
			# calculate teacher predictions
			teacher_x = batch["imgs"].to("cuda")
			teacher_y = batch["labels"].to("cuda")
			reduced_size = [s[:2] for s in batch["imgs_reduced_shape"]]

			hidden_predict = None
			cache = None

			raw_features = dan_manager.models["encoder"](teacher_x)
			features_size = raw_features.size()
			b, c, h, w = features_size

			pos_features = dan_manager.models["decoder"].features_updater.get_pos_features(raw_features)
			features = torch.flatten(pos_features, start_dim=2).permute(2, 0, 1)
			enhanced_features = pos_features
			enhanced_features = torch.flatten(enhanced_features, start_dim=2).permute(2, 0, 1)
			_, teacher_pred, hidden_predict, cache, _ = dan_manager.models["decoder"](features, enhanced_features, teacher_y[:, :-1],
																			  reduced_size, [600]*b, features_size, start=0, hidden_predict=hidden_predict, 
																			  cache=cache, keep_all_weights=True)

			# calculate student predictions
			student_input = [resize_with_padding(image, 640, 480) for image in batch["imgs"]]
			student_input = image_processor(student_input, return_tensors="pt").to("cuda")
			student_output = model(**student_input, labels=batch["labels"].to("cuda"))
			student_loss = student_output.loss

			# calculate distillation loss
			target_vocab_dim = min(teacher_pred.shape[1], student_output.shape[2])
			teacher_prob = teacher_pred.permute(0, 2, 1)[:, :, :target_vocab_dim]
			student_prob = student_output.logits[:, 1:, :target_vocab_dim]
			
			target_seq_length = min(teacher_prob.shape[1], student_prob.shape[1])
			teacher_prob = teacher_prob[:, :target_seq_length, :]
			student_prob = student_prob[:, :target_seq_length, :]

			teacher_prob = F.softmax(teacher_prob, dim=-1)
			student_prob = F.log_softmax(student_prob, dim=-1)
			distill_loss = kl_loss(student_prob, teacher_prob)

			# calculate total loss
			total_loss = (1-distill_alpha)*student_loss + distill_alpha * distill_loss
			total_loss.backward()
			optimizer.step()

		# validation
		model.eval()
		for batch in valid_loader:
			with torch.no_grad():
				# calculate teacher predictions
				teacher_x = batch["imgs"].to("cuda")
				teacher_y = batch["labels"].to("cuda")
				reduced_size = [s[:2] for s in batch["imgs_reduced_shape"]]

				hidden_predict = None
				cache = None

				raw_features = dan_manager.models["encoder"](teacher_x)
				features_size = raw_features.size()
				b, c, h, w = features_size

				pos_features = dan_manager.models["decoder"].features_updater.get_pos_features(raw_features)
				features = torch.flatten(pos_features, start_dim=2).permute(2, 0, 1)
				enhanced_features = pos_features
				enhanced_features = torch.flatten(enhanced_features, start_dim=2).permute(2, 0, 1)
				_, teacher_pred, hidden_predict, cache, _ = dan_manager.models["decoder"](features, enhanced_features, teacher_y[:, :-1],
																				  reduced_size, [600]*b, features_size, start=0, hidden_predict=hidden_predict, 
																				  cache=cache, keep_all_weights=True)

				# calculate student predictions
				student_input = [resize_with_padding(image, 640, 480) for image in batch["imgs"]]
				student_input = image_processor(student_input, return_tensors="pt").to("cuda")
				student_output = model(**student_input, labels=batch["labels"].to("cuda"))
				student_loss = student_output.loss

				# calculate distillation loss
				target_vocab_dim = min(teacher_pred.shape[1], student_output.shape[2])
				teacher_prob = teacher_pred.permute(0, 2, 1)[:, :, :target_vocab_dim]
				student_prob = student_output.logits[:, 1:, :target_vocab_dim]
				
				target_seq_length = min(teacher_prob.shape[1], student_prob.shape[1])
				teacher_prob = teacher_prob[:, :target_seq_length, :]
				student_prob = student_prob[:, :target_seq_length, :]

				teacher_prob = F.softmax(teacher_prob, dim=-1)
				student_prob = F.log_softmax(student_prob, dim=-1)
				distill_loss = kl_loss(student_prob, teacher_prob)

				# calculate metrics
				generated_student_output = model.generate(**student_input, return_dict_in_generate=True, output_scores=True)
				str_x = [LM_ind_to_str(training_dataset.charset, t, oov_symbol="") for t in generated_student_output["sequences"]]
				str_y = batch["raw_labels"]
				stacked_scores = F.softmax(torch.stack(generated_student_output["scores"].permute(1, 2, 0), dim=1))
				confidence_score = [torch.max(stacked_scores[i, :, :], dim=0).values.to("cpu") for i in range(stacked_scores.shape[0])]
				values = {
					"nb_samples": generated_student_output["sequences"].shape[0],
					"str_x": str_x,
					"str_y": str_y,
					"confidence_score": confidence_score,
				}
				batch_metrics = validation_metric_manager.compute_metrics(values, ["cer", "wer", "map_cer", "loer"])
				validation_metric_manager.update_metrics(batch_metrics)
				display_values = validation_metric_manager.get_display_values()

				# log metrics
				metrics_log = f"Epoch {epoch}, CER: {display_values['cer'].item()}, WER: {display_values['wer'].item()}, MAP_CER: {display_values['map_cer'].item()}, LOER: {display_values['loer'].item()}"
				loss_log = f"Student Loss: {student_loss.item()}, Distill Loss: {distill_loss.item()}"
				print(metrics_log)
				print(loss_log)
				if os.path.exists("metrics.log"):
					with open("metrics.log", "a") as f:
						f.write(metrics_log + "\n")
						f.write(loss_log + "\n")
				else:
					with open("metrics.log", "w") as f:
						f.write(metrics_log + "\n")
						f.write(loss_log + "\n")
				
				# save best model
				if display_values["cer"] < best_cer:
					best_cer = display_values["cer"]
					model.save_pretrained("best_model")
					print("Best model saved")
				validation_metric_manager.init_metrics()


