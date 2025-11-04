#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import os
import shutil
import tempfile
from time import time
import typing as t
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from mol_depict.utils.utils_drawing import draw_molecule_rdkit
from mol_depict.utils.utils_generation import get_abbreviations_smiles_mapping
from more_itertools import chunked
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from tqdm import tqdm

from .._const import DATA_PATH
from ..data_modules.data_module import DataModule
from ..datasets.dataset_image import ImageDataset
from .abbreviation_detector import \
    AbbreviationDetectorCPU, AbbreviationDetectorGPU, SpellingCorrector
from .graph_recognizer import \
    GraphRecognizer, StereochemistryRecognizer
from ..utils.utils_dataset import get_bonds_sizes
from ..utils.utils_logging import count_model_parameters

os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
torch.set_float32_matmul_precision("medium")


@dataclass()
class MolgrapherModel:
    # Force all processing to run on CPU.
    force_cpu: bool = field(default=False)
    # Disable multiprocessing for PaddleOCR.
    force_no_multiprocessing: bool = field(default=True)
    # Number of threads used by PyTorch.
    num_threads_pytorch: int = field(default=10)
    # Number of processes for multiprocessing.
    num_processes_mp: int = field(default=10)
    # Number of images processed per batch.
    chunk_size: int = field(default=200)
    # Embed stereochemistry in MOL file output
    # (only valid if a stereo model is used).
    assign_stereo: bool = field(default=True)
    # Embed 2D atom coordinates in output MOL files.
    align_rdkit_output: bool = field(default=False)
    # Apply caption removal preprocessing using PaddleOCR.
    remove_captions: bool = field(default=True)
    # Output folder for saving MOL files.
    save_mol_folder: t.Optional[Path] = field(default=None)
    # Run prediction.
    predict: bool = field(default=True)
    # Apply preprocessing to input images.
    preprocess: bool = field(default=True)
    # Clean specified output folders before running.
    clean: bool = field(default=True)
    # Visualize predicted graphs and MOL prediction.
    visualize: bool = field(default=True)
    # Visualize MOL prediction alone.
    visualize_rdkit: bool = field(default=False)
    # Select between
    # "gc_no_stereo_model", "gc_gcn_model" and "gc_stereo_model".
    # ---
    # "gc_no_stereo_model" has the best accuracy in most cases.
    # "gc_gcn_model" has better accuracy in some cases
    #       but do not recognizes abbreviations.
    # "gc_stereo_model" can recognize stereo-chemistry.
    node_classifier_variant: str = field(default="gc_no_stereo_model")
    visualize_output_folder_path: Path = field(
        # default=DATA_PATH / "visualization/predictions/default/"
        default=Path("./visualizaiton/predictions/default/")
    )
    visualize_rdkit_output_folder_path: Path = field(
        default=DATA_PATH / "visualization/predictions/default_rdkit/"
    )
    config_dataset_graph_path: Path = field(
        default=DATA_PATH / "config_dataset_graph_2.json"
    )
    config_training_graph_path: Path = field(
        default=DATA_PATH / "config_training_graph.json"
    )
    config_dataset_keypoint_path: Path = field(
        default=DATA_PATH / "config_dataset_keypoint.json"
    )
    config_training_keypoint_path: Path = field(
        default=DATA_PATH / "config_training_keypoint.json"
    )

    def __post_init__(self):
        # self.set_default_args()
        # self.args.update(args)

        # print("Arguments:")
        # pprint(self.args)

        # Create save folders
        if self.visualize:
            if self.clean and (
                self.visualize_output_folder_path.exists()
            ):
                shutil.rmtree(self.visualize_output_folder_path)
            self.visualize_output_folder_path\
                .mkdir(parents=True, exist_ok=True)
        if self.predict and (self.save_mol_folder is not None):
            if self.clean and self.save_mol_folder.exists():
                shutil.rmtree(self.save_mol_folder)
            self.save_mol_folder.mkdir(parents=True, exist_ok=True)

        # Automatically set CPU/GPU device
        if not self.force_cpu:
            self.force_cpu = not torch.cuda.is_available()
        device = ('gpu' if not self.force_cpu else 'cpu')
        print(f"PyTorch device: {device}")

        # Read config file
        with open(self.config_dataset_graph_path) as file:
            self.config_dataset_graph = json.load(file)
        with open(self.config_training_graph_path) as file:
            self.config_training_graph = json.load(file)
        with open(self.config_dataset_keypoint_path) as file:
            self.config_dataset_keypoint = json.load(file)
        with open(self.config_training_keypoint_path) as file:
            self.config_training_keypoint = json.load(file)

        # Update config
        self.config_dataset_graph["num_processes_mp"] = \
            self.num_processes_mp
        self.config_dataset_graph["num_threads_pytorch"] = \
            self.num_threads_pytorch
        self.config_dataset_keypoint["num_processes_mp"] = \
            self.num_processes_mp
        self.config_dataset_keypoint["num_threads_pytorch"] = \
            self.num_threads_pytorch

        # Update number of atoms/bonds classes
        # if a node classifier variant is selected.
        if self.node_classifier_variant != "":
            # print("node_classifier_variant", self.node_classifier_variant)
            self.config_model_graph = {}
            self.config_model_graph["node_classifier_variant"] = \
                self.node_classifier_variant
            match self.node_classifier_variant:
                case "gc_no_stereo_model":
                    self.config_dataset_graph["nb_atoms_classes"] = 182
                    self.config_dataset_graph["nb_bonds_classes"] = 6
                    self.config_model_graph["gcn_on"] = False
                case "gc_stereo_model":
                    self.config_dataset_graph["nb_atoms_classes"] = 182
                    self.config_dataset_graph["nb_bonds_classes"] = 8
                    self.config_model_graph["gcn_on"] = False
                case "gc_gcn_model":
                    self.config_dataset_graph["nb_atoms_classes"] = 141
                    self.config_dataset_graph["nb_bonds_classes"] = 5
                    self.config_model_graph["gcn_on"] = True

        # Set # threads
        torch.set_num_threads(self.config_dataset_graph["num_threads_pytorch"])

        # Read model
        self.model = GraphRecognizer(
            self.config_dataset_keypoint,
            self.config_training_keypoint,
            self.config_dataset_graph,
            self.config_training_graph,
            self.config_model_graph,
        )
        kp_det_nparam = \
            round(
                count_model_parameters(self.model.keypoint_detector)/10**6,
                4
            )
        print(
            f"Keypoint detector number parameters: {kp_det_nparam} M"
        )
        node_cla_nparam = \
            round(count_model_parameters(self.model.graph_classifier)/10**6, 4)
        print(
            f"Node classifier number parameters: {node_cla_nparam} M"
        )

        # Set up trainer
        if self.force_cpu:
            self.trainer = pl.Trainer(
                accelerator="cpu",
                precision=self.config_training_graph["precision"],
                logger=False,
            )
        else:
            self.trainer = pl.Trainer(
                accelerator=self.config_training_graph["accelerator"],
                devices=self.config_training_graph["devices"],
                precision=self.config_training_graph["precision"],
                logger=False,
            )

        # Setup abbreviation detector
        if self.force_cpu or (
            self.config_training_graph["accelerator"] == "cpu"
        ):
            self.abbreviation_detector = AbbreviationDetectorCPU(
                self.config_dataset_graph,
                force_cpu=self.force_cpu,
                force_no_multiprocessing=self.force_no_multiprocessing,
            )
        else:
            self.abbreviation_detector = AbbreviationDetectorGPU(
                self.config_dataset_graph,
                force_cpu=self.force_cpu,
                force_no_multiprocessing=self.force_no_multiprocessing,
            )

        # Setup stereochemistry recognizer
        self.stereochemistry_recognizer = \
            StereochemistryRecognizer(self.config_dataset_graph)

        # Set abbreviations list
        with open(
            DATA_PATH / "ocr_mapping/ocr_atoms_classes_mapping.json"
        ) as file:
            self.ocr_atoms_classes_mapping = json.load(file)

        self.abbreviations_smiles_mapping = get_abbreviations_smiles_mapping()
        self.spelling_corrector = \
            SpellingCorrector(self.abbreviations_smiles_mapping)

    def predict_batch(self, _images_paths):
        annotations_batch = []
        for _batch_images_paths in \
                chunked(_images_paths, self.chunk_size):
            annotations_batch.extend(self.predict_single(_batch_images_paths))
        return annotations_batch

    def predict_single(self, images_or_paths: list[str] | str):
        if not isinstance(images_or_paths, list):
            images_or_paths = [images_or_paths]

        # Read dataset images
        data_module = DataModule(
            self.config_dataset_graph,
            dataset_class=ImageDataset,
            images_or_paths=images_or_paths,
            force_cpu=self.force_cpu,
            remove_captions=self.remove_captions,
        )
        data_module.setup_images_benchmarks()
        if self.preprocess:
            print("Starting Caption Removal Preprocessing")
            ref_t = time()
            data_module.preprocess()
            print(
                "Caption Removal Preprocessing completed in ",
                round(time() - ref_t, 2)
            )

        # Get predictions
        print("Starting Keypoint Detection + Node Classification")
        ref_t = time()
        predictions_out = self.trainer.predict(
            self.model,
            dataloaders=data_module.predict_dataloader()
        )
        del data_module  # Release memory

        if predictions_out is None:
            predictions_out = []
        print(
            "Keypoint Detection + Node Classification completed in ",
            round(time() - ref_t, 2)
        )

        images_filenames: list[str] = []
        images_ = []
        predictions = {"graphs": [], "keypoints": [], "confidences": []}
        for _ in range(len(predictions_out)):
            _prediction = predictions_out.pop(0)
            if _prediction is None:
                continue
            for _elem in _prediction["predictions_batch"]["graphs"]:
                predictions["graphs"].append(_elem)
            for _elem in _prediction["predictions_batch"]["keypoints_batch"]:
                predictions["keypoints"].append(_elem)
            for _elem in _prediction["predictions_batch"]["confidences"]:
                predictions["confidences"].append(_elem)
            for _elem in _prediction["batch"]["images_filenames"]:
                images_filenames.append(_elem)
            for _elem in _prediction["batch"]["images"]:
                images_.append(_elem)

        scaling_factor = (
            self.config_dataset_keypoint["image_size"][1]
            // self.config_dataset_keypoint["mask_size"][1]
        )

        # Compute bond size
        bonds_sizes = get_bonds_sizes(predictions["keypoints"], scaling_factor)

        # Recognize abbreviations
        print("Starting Abbreviation Recognition")
        ref_t = time()
        abbreviations_list = self.abbreviation_detector.mp_run(
            images_filenames, predictions["graphs"], bonds_sizes, filter=False
        )
        abbreviations_list_ocr = copy.deepcopy(abbreviations_list)
        print(
            "Abbreviation Recognition completed in ",
            round(time() - ref_t, 2)
        )

        # Recognize stereochemistry
        if self.assign_stereo \
                and self.node_classifier_variant == "gc_stereo_model":
            print("Starting Stereochemistry Recognition")
            ref_t = time()
            predictions["graphs"] = self.stereochemistry_recognizer(
                images_, predictions["graphs"], bonds_sizes
            )
            print(
                "Stereochemistry Recognition completed in ",
                round(time() - ref_t, 2)
            )

        # Create RDKit graph
        print("Starting Graph creation")
        ref_t = time()
        predicted_molecules = []
        for abbreviations, graph, p in zip(
            abbreviations_list, predictions["graphs"], images_filenames
        ):
            predicted_molecule = graph.to_rdkit(
                abbreviations,
                self.abbreviations_smiles_mapping,
                self.ocr_atoms_classes_mapping,
                self.spelling_corrector,
                assign_stereo=(
                    self.assign_stereo
                    and self.node_classifier_variant == "gc_stereo_model"
                ),
                align_rdkit_output=self.align_rdkit_output,
                postprocessing_flags={},
            )
            predicted_molecules.append(predicted_molecule)

        print(f"Graph creation completed in {round(time() - ref_t, 2)}")
        predictions["molecules"] = predicted_molecules

        # Convert to SMILES and set confidence
        predictions["smiles"] = []
        for i, (predicted_molecule, image_filename) in enumerate(
            zip(predictions["molecules"], images_filenames)
        ):
            if self.save_mol_folder is not None:
                molecule_path = self.save_mol_folder / (
                        image_filename
                        .split("/")[-1][:-4]
                        .replace("_preprocessed", "")
                        + ".mol"
                    )
                rdmolfiles.MolToMolFile(
                    predicted_molecule, molecule_path, kekulize=False
                )
            smiles = Chem.MolToSmiles(predicted_molecule)
            if smiles:
                predictions["smiles"].append(smiles)
                if smiles == "C":
                    predictions["confidences"][i] = 0
            else:
                predictions["smiles"].append(None)
                predictions["confidences"][i] = 0
                print("The molecule can not be converted to a valid SMILES")

        # Save annotations
        annotations = []
        for (
            predicted_smiles,
            confidence,
            image_filename,
            abbreviations,
            abbreviations_ocr,
        ) in zip(
            predictions["smiles"],
            predictions["confidences"],
            images_filenames,
            abbreviations_list,
            abbreviations_list_ocr,
        ):
            if predicted_smiles is not None:
                if abbreviations != []:
                    abbreviations_texts = [
                        abbreviation["text"]
                        for abbreviation in abbreviations
                    ]
                else:
                    abbreviations_texts = []
                if abbreviations_ocr != []:
                    abbreviations_ocr_texts = [
                        abbreviation["text"]
                        for abbreviation in abbreviations_ocr
                    ]
                else:
                    abbreviations_ocr_texts = []

                annotation = {
                    "smi": predicted_smiles,
                    "abbreviations": abbreviations_texts,
                    "abbreviations_ocr": abbreviations_ocr_texts,
                    "conf": confidence,
                    "file-info": {"filename": image_filename, "image_nbr": 1},
                    "annotator": {"version": "1.0.0", "program": "MolGrapher"},
                }
                annotations.append(annotation)

            if self.save_mol_folder is not None:
                annotation_filename = self.save_mol_folder / "smiles.jsonl"
                with open(annotation_filename, "a") as f:
                    json.dump(annotation, f)
                    f.write("\n")

        if self.save_mol_folder is not None:
            print("Annotation:")
            print(pd.read_json(path_or_buf=annotation_filename, lines=True))

        # Visualize predictions
        if self.visualize:
            for image_filename, image, graph, keypoints, molecule in tqdm(
                zip(
                    images_filenames,
                    images_,
                    predictions["graphs"],
                    predictions["keypoints"],
                    predictions["molecules"],
                ),
                total=len(images_filenames),
            ):
                smiles = Chem.MolToSmiles(molecule)
                if smiles != "C":
                    figure, axis = plt.subplots(1, 3, figsize=(20, 10))
                else:
                    figure, axis = plt.subplots(1, 2, figsize=(20, 10))
                axis[0].imshow(image.permute(1, 2, 0))

                axis[0].scatter(
                    [
                        (keypoint[0] * scaling_factor + scaling_factor // 2)
                        for keypoint in keypoints
                    ],
                    [
                        (keypoint[1] * scaling_factor + scaling_factor // 2)
                        for keypoint in keypoints
                    ],
                    color="red",
                    alpha=0.5,
                )

                graph.display_data_nodes_only(axis=axis[1])

                if smiles != "C":
                    image = draw_molecule_rdkit(
                        smiles=smiles,
                        molecule=molecule,
                        augmentations=False,
                        path=tempfile.NamedTemporaryFile(
                            suffix=".png", delete=False
                        ).name,
                    )
                    if image is not None:
                        axis[2].imshow(image.permute(1, 2, 0))
                fig_path = \
                    self.visualize_output_folder_path / \
                    image_filename.split('/')[-1]
                print(fig_path)
                plt.savefig(fig_path)
                plt.close()

        if self.visualize_rdkit:
            # for image_filename in self.input_images_paths:
            for image_filename in images_or_paths:
                if self.save_mol_folder is None:
                    raise AttributeError

                molecule_path = (
                    self.save_mol_folder
                    / (
                        image_filename.split("/")[-1][:-4]
                        .replace("_preprocessed", "")
                        + ".mol"
                    )
                )
                if molecule_path.exists():
                    print(molecule_path)
                    image = Image.open(image_filename).convert("RGB")
                    figure, axis = plt.subplots(1, 2, figsize=(20, 10))
                    axis[0].imshow(image)
                    molecule = \
                        rdmolfiles.MolFromMolFile(
                            molecule_path,
                            sanitize=False
                        )
                    image = draw_molecule_rdkit(
                        smiles=Chem.MolToSmiles(molecule),
                        molecule=molecule,
                        augmentations=False,
                        path=tempfile.NamedTemporaryFile(
                            suffix=".png", delete=False
                        ).name,
                    )
                    if image is not None:
                        axis[1].imshow(image.permute(1, 2, 0))
                    output_path = \
                        self.visualize_rdkit_output_folder_path / \
                        image_filename.split('/')[-1]
                    plt.savefig(output_path)
                plt.close()

        return annotations
