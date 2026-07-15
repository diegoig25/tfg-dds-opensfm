# pyre-strict
"""Tools to extract features."""

import logging
import threading
import time
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
import torch
from opensfm.dds_2 import DDSAutoencoder, DDSConfig
import cv2
import numpy as np
from numpy.typing import NDArray
from opensfm import context, pyfeatures


logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# DDS singleton (cargado una sola vez por proceso)
# ============================================================
# Lock para evitar race condition si OpenSfM paraleliza con threads/joblib.
# Mucho más importante: el modelo se debe construir y cargar UNA sola vez
# por proceso, no en cada llamada a extract_features_dds().
_DDS_MODEL: Optional[DDSAutoencoder] = None
_DDS_DEVICE: Optional[torch.device]  = None
_DDS_LOCK = threading.Lock()

# Estadísticas ImageNet — DEBEN coincidir con train_dds.py.
# Si entrenaste con otras stats, cambia estos valores.
_DDS_IMAGENET_MEAN = torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1)  # entrada [0,1], sin normalizar
_DDS_IMAGENET_STD  = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)  # entrada [0,1], sin normalizar


class SemanticData:
    segmentation: NDArray
    instances: Optional[NDArray]
    labels: List[Dict[str, Any]]

    def __init__(
        self,
        segmentation: NDArray,
        instances: Optional[NDArray],
        labels: List[Dict[str, Any]],
    ) -> None:
        self.segmentation = segmentation
        self.instances = instances
        self.labels = labels

    def has_instances(self) -> bool:
        return self.instances is not None

    def mask(self, mask: NDArray) -> "SemanticData":
        try:
            segmentation = self.segmentation[mask]
            instances = self.instances
            if instances is not None:
                instances = instances[mask]
        except IndexError:
            logger.error(
                f"Invalid mask array of dtype {mask.dtype}, shape {mask.shape}: {mask}"
            )
            raise

        return SemanticData(segmentation, instances, self.labels)


class FeaturesData:
    points: NDArray
    descriptors: Optional[NDArray]
    colors: NDArray
    semantic: Optional[SemanticData]
    depths: Optional[NDArray]  # New field. This field is not serialized yet

    FEATURES_VERSION: int = 3
    FEATURES_HEADER: str = "OPENSFM_FEATURES_VERSION"

    def __init__(
        self,
        points: NDArray,
        descriptors: Optional[NDArray],
        colors: NDArray,
        semantic: Optional[SemanticData],
        depths: Optional[NDArray] = None,
    ) -> None:
        self.points = points
        self.descriptors = descriptors
        self.colors = colors
        self.semantic = semantic
        self.depths = depths

    def get_segmentation(self) -> Optional[NDArray]:
        semantic = self.semantic
        if not semantic:
            return None
        if semantic.segmentation is not None:
            return semantic.segmentation
        return None

    def has_instances(self) -> bool:
        semantic = self.semantic
        if not semantic:
            return False
        return semantic.instances is not None

    def mask(self, mask: NDArray) -> "FeaturesData":
        if self.semantic:
            masked_semantic = self.semantic.mask(mask)
        else:
            masked_semantic = None
        return FeaturesData(
            self.points[mask],
            self.descriptors[mask] if self.descriptors is not None else None,
            self.colors[mask],
            masked_semantic,
            self.depths[mask] if self.depths is not None else None,
        )

    def save(self, fileobject: Union[str, BinaryIO], config: Dict[str, Any]) -> None:
        """Save features from file (path like or file object like)"""
        feature_type = config["feature_type"].upper()
        if (
            (
                feature_type == "AKAZE"
                and config["akaze_descriptor"] in ["MLDB_UPRIGHT", "MLDB"]
            )
            or (feature_type == "HAHOG" and config["hahog_normalize_to_uchar"])
            or (feature_type == "ORB")
        ):
            feature_data_type = np.uint8
        else:
            feature_data_type = np.float32
        descriptors = self.descriptors
        if descriptors is None:
            raise RuntimeError("No descriptors found, cannot save features data.")
        semantic = self.semantic
        if semantic:
            instances = semantic.instances
            np.savez_compressed(
                fileobject,
                points=self.points.astype(np.float32),
                descriptors=descriptors.astype(feature_data_type),
                colors=self.colors,
                segmentations=semantic.segmentation.astype(np.uint8),
                instances=instances.astype(np.int16) if instances is not None else [],
                segmentation_labels=np.array(semantic.labels).astype(str),
                OPENSFM_FEATURES_VERSION=self.FEATURES_VERSION,
            )
        else:
            np.savez_compressed(
                fileobject,
                points=self.points.astype(np.float32),
                descriptors=descriptors.astype(feature_data_type),
                colors=self.colors,
                segmentations=[],
                instances=[],
                segmentation_labels=[],
                OPENSFM_FEATURES_VERSION=self.FEATURES_VERSION,
            )

    @classmethod
    def from_file(
        cls, fileobject: Union[str, BinaryIO], config: Dict[str, Any]
    ) -> "FeaturesData":
        """Load features from file (path like or file object like)"""
        s = np.load(fileobject, allow_pickle=False)
        version = cls._features_file_version(s)
        return getattr(cls, "_from_file_v%d" % version)(s, config)

    @classmethod
    def _features_file_version(cls, obj: Dict[str, Any]) -> int:
        """Retrieve features file version. Return 0 if none"""
        if cls.FEATURES_HEADER in obj:
            return obj[cls.FEATURES_HEADER]
        else:
            return 0

    @classmethod
    def _from_file_v0(
        cls, data: Dict[str, NDArray], config: Dict[str, Any]
    ) -> "FeaturesData":
        """Base version of features file

        Scale (desc[2]) set to reprojection_error_sd by default (legacy behaviour)
        """
        feature_type = config["feature_type"]
        if feature_type == "HAHOG" and config["hahog_normalize_to_uchar"]:
            descriptors = data["descriptors"].astype(np.float32)
        else:
            descriptors = data["descriptors"]
        points = data["points"]
        points[:, 2:3] = config["reprojection_error_sd"]
        return FeaturesData(points, descriptors, data["colors"].astype(float), None)

    @classmethod
    def _from_file_v1(
        cls, data: Dict[str, NDArray], config: Dict[str, Any]
    ) -> "FeaturesData":
        """Version 1 of features file

        Scale is not properly set higher in the pipeline, default is gone.
        """
        feature_type = config["feature_type"]
        if feature_type == "HAHOG" and config["hahog_normalize_to_uchar"]:
            descriptors = data["descriptors"].astype(np.float32)
        else:
            descriptors = data["descriptors"]
        return FeaturesData(
            data["points"], descriptors, data["colors"].astype(float), None
        )

    @classmethod
    def _from_file_v2(
        cls,
        data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> "FeaturesData":
        """
        Version 2 of features file

        Added segmentation, instances and segmentation labels. This version has been introduced at
        e5da878bea455a1e4beac938cb30b796acfe3c8c, but has been superseded by version 3 as this version
        uses 'allow_pickle=True' which isn't safe (RCE vulnerability)
        """
        feature_type = config["feature_type"]
        if feature_type == "HAHOG" and config["hahog_normalize_to_uchar"]:
            descriptors = data["descriptors"].astype(np.float32)
        else:
            descriptors = data["descriptors"]

        # luckily, because os lazy loading, we can still load 'segmentations' and 'instances' ...
        pickle_message = (
            "Cannot load {} as these were generated with "
            "version 2 which isn't supported anymore because of RCE vulnerablity."
            "Please consider re-extracting features data for this dataset"
        )
        try:
            has_segmentation = (data["segmentations"] != None).all()
            has_instances = (data["instances"] != None).all()
        except ValueError:
            logger.warning(pickle_message.format("segmentations and instances"))
            has_segmentation, has_instances = False, False

        # ... whereas 'labels' can't be loaded anymore, as it is a plain 'list' object. Not an
        # issue since these labels are used for description only and not actual filtering.
        try:
            labels = data["segmentation_labels"]
        except ValueError:
            logger.warning(pickle_message.format("labels"))
            labels = []

        if has_segmentation or has_instances:
            semantic_data = SemanticData(
                data["segmentations"] if has_segmentation else None,
                data["instances"] if has_instances else None,
                labels,
            )
        else:
            semantic_data = None
        return FeaturesData(
            data["points"], descriptors, data["colors"].astype(float), semantic_data
        )

    @classmethod
    def _from_file_v3(
        cls,
        data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> "FeaturesData":
        """
        Version 3 of features file

        Same as version 2, except that
        """
        feature_type = config["feature_type"]
        if feature_type == "HAHOG" and config["hahog_normalize_to_uchar"]:
            descriptors = data["descriptors"].astype(np.float32)
        else:
            descriptors = data["descriptors"]

        has_segmentation = len(data["segmentations"]) > 0
        has_instances = len(data["instances"]) > 0

        if has_segmentation or has_instances:
            semantic_data = SemanticData(
                data["segmentations"] if has_segmentation else None,
                data["instances"] if has_instances else None,
                data["segmentation_labels"],
            )
        else:
            semantic_data = None
        return FeaturesData(
            data["points"], descriptors, data["colors"].astype(float), semantic_data
        )


def resized_image(image: NDArray, max_size: int) -> NDArray:
    """Resize image to feature_process_size."""
    h, w = image.shape[:2]
    size = max(w, h)
    if 0 < max_size < size:
        dsize = w * max_size // size, h * max_size // size
        return cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_AREA)
    else:
        return image


def root_feature(desc: NDArray, l2_normalization: bool = False) -> NDArray:
    if l2_normalization:
        s2 = np.linalg.norm(desc, axis=1)
        desc = (desc.T / s2).T
    s = np.sum(desc, 1)
    desc = np.sqrt(desc.T / s).T
    return desc


def root_feature_surf(
    desc: NDArray, l2_normalization: bool = False, partial: bool = False
) -> NDArray:
    """
    Experimental square root mapping of surf-like feature, only work for 64-dim surf now
    """
    if desc.shape[1] == 64:
        if l2_normalization:
            s2 = np.linalg.norm(desc, axis=1)
            desc = (desc.T / s2).T
        if partial:
            ii = np.array([i for i in range(64) if (i % 4 == 2 or i % 4 == 3)])
        else:
            ii = np.arange(64)
        desc_sub = np.abs(desc[:, ii])
        desc_sub_sign = np.sign(desc[:, ii])
        # s_sub = np.sum(desc_sub, 1)  # This partial normalization gives slightly better results for AKAZE surf
        s_sub = np.sum(np.abs(desc), 1)
        desc_sub = np.sqrt(desc_sub.T / s_sub).T
        desc[:, ii] = desc_sub * desc_sub_sign
    return desc


def normalized_image_coordinates(
    pixel_coords: NDArray, width: int, height: int
) -> NDArray:
    size = max(width, height)
    p = np.empty((len(pixel_coords), 2))
    p[:, 0] = (pixel_coords[:, 0] + 0.5 - width / 2.0) / size
    p[:, 1] = (pixel_coords[:, 1] + 0.5 - height / 2.0) / size
    return p


def denormalized_image_coordinates(
    norm_coords: NDArray, width: int, height: int
) -> NDArray:
    size = max(width, height)
    p = np.empty((len(norm_coords), 2))
    p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
    p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0
    return p


def normalize_features(
    points: NDArray, desc: NDArray, colors: NDArray, width: int, height: int
) -> Tuple[
    NDArray,
    NDArray,
    NDArray,
]:
    """Normalize feature coordinates and size."""
    points[:, :2] = normalized_image_coordinates(points[:, :2], width, height)
    points[:, 2:3] /= max(width, height)
    return points, desc, colors


def _in_mask(point: NDArray, width: int, height: int, mask: NDArray) -> bool:
    """Check if a point is inside a binary mask."""
    u = mask.shape[1] * (point[0] + 0.5) / width
    v = mask.shape[0] * (point[1] + 0.5) / height
    return mask[int(v), int(u)] != 0


def extract_features_sift(
    image: NDArray, config: Dict[str, Any], features_count: int
) -> Tuple[NDArray, NDArray]:
    sift_edge_threshold = config["sift_edge_threshold"]
    sift_peak_threshold = float(config["sift_peak_threshold"])
    sift_nfeatures = config["sift_nfeatures"]
    sift_octave_layers = config["sift_octave_layers"]
    sift_sigma = float(config["sift_sigma"])
    while True:
        logger.debug("Computing sift with threshold {0}".format(sift_peak_threshold))
        t = time.time()
        # SIFT support is in cv2 main from version 4.4.0
        if context.OPENCV44 or context.OPENCV5:
            detector = cv2.SIFT_create(
                nfeatures=sift_nfeatures,
                nOctaveLayers=sift_octave_layers,
                contrastThreshold=sift_peak_threshold,
                edgeThreshold=sift_edge_threshold,
                sigma=sift_sigma,
            )
            descriptor = detector
        elif context.OPENCV3:
            detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=sift_nfeatures,
                nOctaveLayers=sift_octave_layers,
                contrastThreshold=sift_peak_threshold,
                edgeThreshold=sift_edge_threshold,
                sigma=sift_sigma,
            )
            descriptor = detector
        else:
            detector = cv2.FeatureDetector_create("SIFT")
            descriptor = cv2.DescriptorExtractor_create("SIFT")
            detector.setDouble("edgeThreshold", sift_edge_threshold)

        points = detector.detect(image)
        logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))
        if len(points) < features_count and sift_peak_threshold > 0.0001:
            sift_peak_threshold = (sift_peak_threshold * 2) / 3
            logger.debug("reducing threshold")
        else:
            logger.debug("done")
            break

    points, desc = descriptor.compute(image, points)
    if config["feature_root"]:
        desc = root_feature(desc)
        points = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
    else:
        points = np.array(np.zeros((0, 3)))
        desc = np.array(np.zeros((0, 3)))
    return points, desc


# ============================================================
# DDS — singleton + extracción
# ============================================================
# Integración del modelo DDS preentrenado dentro del pipeline de detección
# de features de OpenSfM.
#
# Puntos críticos:
#   #1  Normalización del entrenamiento aplicada en inferencia (sin esto
#       el modelo ve una distribución distinta a la de train).
#   #2  in_channels (1 o 3) leído del checkpoint, no hardcodeado:
#       el modelo decide qué espera, features.py se adapta.
#   #3  Si OpenSfM ya convirtió la imagen a gray, replicamos canales o
#       mantenemos 1 canal según el modelo.
#   #4  original_hw pasado a dds_to_opensfm() para coordenadas en píxeles
#       correctas.
#   #5  load_state_dict(strict=True): falla en voz alta si los pesos no
#       encajan con la arquitectura.
#   #6  cfg leído del checkpoint cuando está disponible → la arquitectura
#       del modelo construido coincide EXACTAMENTE con la del entrenamiento.
#   #7  Lock para evitar race condition al modificar cfg.max_points
#       (joblib paraleliza la extracción de features).

def _get_dds_model(config: Dict[str, Any]) -> Tuple[DDSAutoencoder, torch.device]:
    """Construye e inicializa el modelo DDS una sola vez por proceso.

    Estrategia para los hiperparámetros de arquitectura:
      1. Si el checkpoint contiene 'cfg' (lo guarda train_dds.py), USARLO.
         Garantiza que la arquitectura coincide exactamente con la del
         entrenamiento, sin depender de que el usuario replique los
         parámetros en config.yaml.
      2. Si no hay cfg en el checkpoint (formato antiguo), construir
         DDSConfig desde config.yaml con los defaults seguros.
    """
    global _DDS_MODEL, _DDS_DEVICE

    if _DDS_MODEL is not None:
        return _DDS_MODEL, _DDS_DEVICE  # type: ignore[return-value]

    with _DDS_LOCK:
        # Re-check tras adquirir el lock (otro thread podría haberlo cargado)
        if _DDS_MODEL is not None:
            return _DDS_MODEL, _DDS_DEVICE  # type: ignore[return-value]

        device_name = config.get(
            "dds_device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        _DDS_DEVICE = torch.device(device_name)

        weights_path = config.get("dds_model_path", None)
        if weights_path is None:
            raise ValueError("DDS requires 'dds_model_path' in config.yaml")

        # ── Alias del módulo dds_2 ─────────────────────────────────
        # El checkpoint guardado por train_dds.py contiene una instancia
        # serializada de DDSConfig referida al módulo "dds_2" (sin prefijo,
        # porque train_dds.py añade opensfm/ al sys.path y hace
        # "from dds_2 import ..."). Aquí dentro de OpenSfM el módulo
        # se importa como "opensfm.dds_2", así que pickle no encuentra
        # "dds_2" y falla con ModuleNotFoundError.
        # Solución: registrar el alias antes de cargar el checkpoint.
        import sys as _sys
        from opensfm import dds_2 as _dds_2_module
        _sys.modules.setdefault("dds_2", _dds_2_module)
        from opensfm import dds_sift as _dds_sift_module
        _sys.modules.setdefault("dds_sift", _dds_sift_module)

        ckpt = torch.load(weights_path, map_location=_DDS_DEVICE, weights_only=False)

        # ── Obtener cfg: del checkpoint si existe, si no del config.yaml ──
        ckpt_cfg = None
        if isinstance(ckpt, dict) and "cfg" in ckpt and ckpt["cfg"] is not None:
            ckpt_cfg = ckpt["cfg"]

        if ckpt_cfg is not None:
            # Usar la arquitectura EXACTA del entrenamiento.
            # Solo sobrescribimos parámetros de RUNTIME (no de arquitectura):
            #   - max_points: el usuario puede pedir más/menos kp por imagen
            #   - nms_kernel, min_score: post-procesado
            #   - default_size, default_angle: solo metadata para OpenSfM
            cfg = DDSConfig(
                in_channels    = ckpt_cfg.in_channels,
                base_channels  = ckpt_cfg.base_channels,
                depth          = ckpt_cfg.depth,
                max_channels   = ckpt_cfg.max_channels,
                descriptor_dim = ckpt_cfg.descriptor_dim,
                max_points     = config.get(
                    "dds_max_points", config.get("feature_min_frames", 4000)
                ),
                nms_kernel     = config.get("dds_nms_kernel", 9),
                min_score      = config.get("dds_min_score", 0.01),
                default_size   = config.get("dds_default_size", 8.0),
                default_angle  = config.get("dds_default_angle", 0.0),
                # Hiperparámetros del paper se mantienen del cfg original
                alpha_l0       = ckpt_cfg.alpha_l0,
                beta           = ckpt_cfg.beta,
                zeta           = ckpt_cfg.zeta,
                gamma          = ckpt_cfg.gamma,
                alpha_l1       = ckpt_cfg.alpha_l1,
                alpha_tau      = ckpt_cfg.alpha_tau,
                alpha_gamma_f  = ckpt_cfg.alpha_gamma_f,
                eps_gamma_f    = ckpt_cfg.eps_gamma_f,
                # Descriptor aprendido: imprescindible para DDS-DDS. getattr con
                # default False → checkpoints viejos (solo detector) cargan igual.
                use_learned_descriptor = getattr(ckpt_cfg, "use_learned_descriptor", False),
                descriptor_hidden      = getattr(ckpt_cfg, "descriptor_hidden", 128),
                descriptor_dilation    = getattr(ckpt_cfg, "descriptor_dilation", 2),
            )
            logger.info("DDS: usando cfg del checkpoint (arquitectura del entrenamiento)")
        else:
            # Fallback: reconstruir desde config.yaml.
            # ATENCIÓN: los parámetros deben coincidir EXACTAMENTE con los
            # del entrenamiento o load_state_dict(strict=True) fallará.
            cfg = DDSConfig(
                in_channels    = config.get("dds_in_channels", 3),
                base_channels  = config.get("dds_base_channels", 32),
                depth          = config.get("dds_depth", 4),
                max_channels   = config.get("dds_max_channels", 256),
                descriptor_dim = config.get("dds_descriptor_dim", 128),
                max_points     = config.get(
                    "dds_max_points", config.get("feature_min_frames", 4000)
                ),
                nms_kernel     = config.get("dds_nms_kernel", 9),
                min_score      = config.get("dds_min_score", 0.01),
                default_size   = config.get("dds_default_size", 8.0),
                default_angle  = config.get("dds_default_angle", 0.0),
            )
            logger.warning(
                "DDS: checkpoint sin cfg embebido; usando parámetros de config.yaml. "
                "Asegúrate de que coinciden con el entrenamiento."
            )

        model = DDSAutoencoder(cfg)

        # Aceptamos checkpoints tipo {"model_state_dict": ...} o state_dict puro
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # strict=True: si los pesos no encajan, fallar en voz alta.
        # Con strict=False se cargan parcialmente y el modelo devuelve basura
        # sin avisar.
        model.load_state_dict(state_dict, strict=True)

        model.to(_DDS_DEVICE)
        model.eval()

        # Cachear constantes de normalización en el device.
        # Si el modelo se entrenó en grayscale, usamos mean/std de un solo canal.
        global _DDS_IMAGENET_MEAN, _DDS_IMAGENET_STD
        if cfg.in_channels == 1:
            _DDS_IMAGENET_MEAN = torch.tensor([0.0]).view(1, 1, 1, 1).to(_DDS_DEVICE)  # [0,1]
            _DDS_IMAGENET_STD  = torch.tensor([1.0]).view(1, 1, 1, 1).to(_DDS_DEVICE)  # [0,1]
        else:
            _DDS_IMAGENET_MEAN = torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1).to(_DDS_DEVICE)  # [0,1]
            _DDS_IMAGENET_STD  = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1).to(_DDS_DEVICE)  # [0,1]

        _DDS_MODEL = model
        logger.info(
            f"DDS model loaded from {weights_path} on {_DDS_DEVICE} "
            f"(in_channels={cfg.in_channels}, base={cfg.base_channels}, "
            f"depth={cfg.depth}, desc_dim={cfg.descriptor_dim})"
        )
        return _DDS_MODEL, _DDS_DEVICE


@torch.no_grad()
def extract_features_dds(
    image: NDArray, config: Dict[str, Any], features_count: int
) -> Tuple[NDArray, NDArray]:
    """
    Extrae keypoints + descriptores con el modelo DDS preentrenado.

    image: imagen ya pre-procesada por OpenSfM (post-resize). Puede llegar como:
       - shape (H, W)        : grayscale 2D
       - shape (H, W, 1)     : grayscale 3D con canal explícito
       - shape (H, W, 3)     : RGB (caso raro, ya que extract_features convierte a gray)
       dtype: np.uint8 [0, 255]

    El modelo DDS espera entrada normalizada con stats ImageNet (si in_channels=3)
    o con stats de un canal (si in_channels=1).
    """
    model, device = _get_dds_model(config)
    in_channels = model.cfg.in_channels

    # Adaptar la imagen al número de canales del modelo entrenado.
    # OpenSfM ya convirtió la imagen original a grayscale en extract_features().
    if in_channels == 3:
        # Modelo entrenado en RGB. Replicamos el canal gray 3 veces.
        # NOTA: perdemos info de color porque OpenSfM ya hizo cvtColor a gray.
        # Si quieres color real, modifica extract_features() para que no
        # convierta antes de llamar a DDS.
        if image.ndim == 2:
            image_in = np.stack([image, image, image], axis=-1)  # (H, W, 3)
        elif image.ndim == 3 and image.shape[2] == 1:
            image_in = np.repeat(image, 3, axis=2)                # (H, W, 3)
        elif image.ndim == 3 and image.shape[2] == 3:
            image_in = image
        else:
            raise ValueError(f"Unexpected image shape for DDS: {image.shape}")
    elif in_channels == 1:
        # Modelo entrenado en grayscale.
        if image.ndim == 2:
            image_in = image[..., None]                           # (H, W, 1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image_in = image
        elif image.ndim == 3 and image.shape[2] == 3:
            # Convertir a gray para que coincida con el régimen de entrenamiento
            image_in = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., None]
        else:
            raise ValueError(f"Unexpected image shape for DDS: {image.shape}")
    else:
        raise ValueError(f"DDS in_channels={in_channels} not supported (must be 1 or 3)")

    if image_in.dtype != np.uint8:
        image_in = image_in.astype(np.uint8)

    h_orig, w_orig = image_in.shape[:2]

    # ── Normalización del entrenamiento ─────────────────────────
    # uint8 → float [0,1] → normalizado con stats del entrenamiento.
    # ESTE ES EL PASO MÁS CRÍTICO. Sin él, el modelo ve una distribución
    # estadística distinta a la de train.
    image_t = torch.from_numpy(image_in).to(device).float() / 255.0
    image_t = image_t.permute(2, 0, 1).unsqueeze(0)             # [1, C, H, W]
    image_t = (image_t - _DDS_IMAGENET_MEAN) / _DDS_IMAGENET_STD

    # ── Lock para max_points temporal ──────────────────────────
    # OpenSfM puede llamar a esta función desde múltiples threads vía joblib.
    # Sin el lock, dos threads pueden pisarse el cfg.max_points.
    with _DDS_LOCK:
        original_max_points = model.cfg.max_points
        model.cfg.max_points = features_count
        try:
            # Pasamos h_orig/w_orig para que dds_to_opensfm() no tenga
            # que adivinar y las coordenadas salgan en píxeles de la imagen
            # recibida.
            points, desc = model.extract_opensfm_features(
                image_t,
                original_hw=(h_orig, w_orig),
            )
        finally:
            model.cfg.max_points = original_max_points

    points = points.astype(np.float32)
    desc = desc.astype(np.float32)

    # ── Deduplicación de keypoints ─────────────────────────────────
    # DDS puede producir múltiples keypoints en posiciones (x, y) idénticas
    # o muy cercanas (sub-pixel). Esto rompe cv2.findFundamentalMat: cuando
    # RANSAC elige 8 puntos al azar, si caen colineales o duplicados, la
    # matriz de 8-point se vuelve singular y OpenCV crashea con el error
    # críptico "rowRange". La SHAPE y dtype son correctos, lo malo es la
    # degeneración geométrica.
    #
    # Solución: redondear coords a entero (precisión píxel suficiente para
    # SfM) y quedarnos solo con el primer keypoint por celda.
    if len(points) > 0:
        keys = np.round(points[:, :2]).astype(np.int32)
        # np.unique con return_index preserva el ORDEN original (= orden por
        # score, ya que dds_to_opensfm devuelve ordenado por score desc).
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)  # restaurar orden original
        n_before = len(points)
        points = points[unique_idx]
        desc = desc[unique_idx]
        if len(points) < n_before:
            logger.debug(
                f"DDS dedupe: {n_before} → {len(points)} keypoints "
                f"({n_before - len(points)} duplicados eliminados)"
            )

    # ── Descriptor SIFT sobre los keypoints de DDS ─────────────────
    # El descriptor DDS está mal definido (no discrimina bien). Por
    # indicación de la tutoría, usamos DDS solo como DETECTOR y calculamos
    # el descriptor con SIFT sobre las posiciones que DDS seleccionó.
    # Se activa con dds_descriptor: SIFT en config.yaml (por defecto: DDS).
    descriptor_type = config.get("dds_descriptor", "DDS").upper()
    if descriptor_type == "SIFT" and len(points) > 0:
        points, desc = _sift_descriptor_on_keypoints(image_in, points, config)
        logger.debug(f"DDS+SIFT: {len(points)} keypoints con descriptor SIFT")

    logger.debug(f"DDS: extracted {len(points)} keypoints")
    return points, desc


def _sift_descriptor_on_keypoints(
    image: NDArray, points: NDArray, config: Dict[str, Any]
) -> Tuple[NDArray, NDArray]:
    """
    Calcula descriptores SIFT en las posiciones de keypoint que DDS detectó.

    image:  imagen (H, W) o (H, W, C) uint8 en el sistema de coords de `points`.
    points: array [N, 4] (x, y, size, angle) de DDS, en píxeles de la imagen.

    Devuelve (points, desc) filtrados a los keypoints para los que SIFT
    pudo calcular descriptor. SIFT puede descartar algún keypoint pegado al
    borde (necesita un parche alrededor), por eso los conteos pueden bajar.

    Nota: el descriptor SIFT es de 128 dimensiones, igual que el DDS, así
    que el resto del pipeline (FLANN, matching) no cambia.
    """
    # SIFT necesita imagen en escala de grises uint8
    if image.ndim == 3:
        if image.shape[2] == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image[:, :, 0]
    else:
        img_gray = image
    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)

    # Construir el extractor SIFT (mismos parámetros que extract_features_sift)
    if context.OPENCV44 or context.OPENCV5:
        sift = cv2.SIFT_create(
            nfeatures=config.get("sift_nfeatures", 0),
            nOctaveLayers=config.get("sift_octave_layers", 3),
            contrastThreshold=float(config.get("sift_peak_threshold", 0.0)),
            edgeThreshold=config.get("sift_edge_threshold", 10),
            sigma=float(config.get("sift_sigma", 1.6)),
        )
    elif context.OPENCV3:
        sift = cv2.xfeatures2d.SIFT_create(
            nfeatures=config.get("sift_nfeatures", 0),
            nOctaveLayers=config.get("sift_octave_layers", 3),
            contrastThreshold=float(config.get("sift_peak_threshold", 0.0)),
            edgeThreshold=config.get("sift_edge_threshold", 10),
            sigma=float(config.get("sift_sigma", 1.6)),
        )
    else:
        raise RuntimeError("SIFT no disponible en esta versión de OpenCV")

    # Convertir los puntos DDS a cv2.KeyPoint.
    # DDS no estima escala ni orientación reales, así que damos un tamaño
    # razonable y dejamos que SIFT calcule la orientación dominante.
    sift_size = float(config.get("dds_sift_keypoint_size", 8.0))
    keypoints = [
        cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=sift_size)
        for p in points
    ]

    # compute() devuelve los keypoints que sobrevivieron + sus descriptores.
    # SIFT puede recalcular la orientación y duplicar un keypoint si detecta
    # varias orientaciones dominantes; usamos los keypoints devueltos.
    kps_out, desc_out = sift.compute(img_gray, keypoints)

    if desc_out is None or len(kps_out) == 0:
        # Ningún descriptor válido (raro). Devolver vacío seguro.
        return np.zeros((0, 4), np.float32), np.zeros((0, 128), np.float32)

    pts_out = np.array(
        [(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in kps_out],
        dtype=np.float32,
    )
    desc_out = desc_out.astype(np.float32)

    # Normalización del descriptor SIFT.
    # root_feature (RootSIFT) si feature_root=true; si no, L2-norm para que
    # sea compatible con el matching FLANN del resto del pipeline.
    if config.get("feature_root", False):
        desc_out = root_feature(desc_out)
    else:
        norms = np.linalg.norm(desc_out, axis=1, keepdims=True)
        desc_out = desc_out / np.maximum(norms, 1e-8)

    return pts_out, desc_out


def akaze_descriptor_type(name: str) -> pyfeatures.AkazeDescriptorType:
    d = pyfeatures.AkazeDescriptorType.__dict__
    if name in d:
        return d[name]
    else:
        logger.debug("Wrong akaze descriptor type")
        return d["MSURF"]


def extract_features_akaze(
    image: NDArray, config: Dict[str, Any], features_count: int
) -> Tuple[NDArray, NDArray]:
    options = pyfeatures.AKAZEOptions()
    options.omax = config["akaze_omax"]
    akaze_descriptor_name = config["akaze_descriptor"]
    options.descriptor = akaze_descriptor_type(akaze_descriptor_name)
    options.descriptor_size = config["akaze_descriptor_size"]
    options.descriptor_channels = config["akaze_descriptor_channels"]
    options.dthreshold = config["akaze_dthreshold"]
    options.kcontrast_percentile = config["akaze_kcontrast_percentile"]
    options.use_isotropic_diffusion = config["akaze_use_isotropic_diffusion"]
    options.target_num_features = features_count
    options.use_adaptive_suppression = config["feature_use_adaptive_suppression"]

    logger.debug("Computing AKAZE with threshold {0}".format(options.dthreshold))
    t = time.time()
    points, desc = pyfeatures.akaze(image, options)
    logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))

    if config["feature_root"]:
        if akaze_descriptor_name in ["SURF_UPRIGHT", "MSURF_UPRIGHT"]:
            desc = root_feature_surf(desc, partial=True)
        elif akaze_descriptor_name in ["SURF", "MSURF"]:
            desc = root_feature_surf(desc, partial=False)
    points = points.astype(float)
    return points, desc


def extract_features_hahog(
    image: NDArray, config: Dict[str, Any], features_count: int
) -> Tuple[NDArray, NDArray]:
    t = time.time()
    points, desc = pyfeatures.hahog(
        image.astype(np.float32) / 255,  # VlFeat expects pixel values between 0, 1
        peak_threshold=config["hahog_peak_threshold"],
        edge_threshold=config["hahog_edge_threshold"],
        target_num_features=features_count,
    )

    if config["feature_root"]:
        desc = np.sqrt(desc)
        uchar_scaling = 362  # x * 512 < 256  =>  sqrt(x) * 362 < 256
    else:
        uchar_scaling = 512

    if config["hahog_normalize_to_uchar"]:
        # pyre-fixme[16]: `int` has no attribute `clip`.
        desc = (uchar_scaling * desc).clip(0, 255).round()

    logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))
    return points, desc


def extract_features_orb(
    image: NDArray, config: Dict[str, Any], features_count: int
) -> Tuple[NDArray, NDArray]:
    if context.OPENCV3:
        detector = cv2.ORB_create(nfeatures=features_count)
        descriptor = detector
    else:
        detector = cv2.FeatureDetector_create("ORB")
        descriptor = cv2.DescriptorExtractor_create("ORB")
        detector.setDouble("nFeatures", features_count)

    logger.debug("Computing ORB")
    t = time.time()
    points = detector.detect(image)

    points, desc = descriptor.compute(image, points)
    if desc is not None:
        points = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
    else:
        points = np.array(np.zeros((0, 3)))
        desc = np.array(np.zeros((0, 3)))

    logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))
    return points, desc


def extract_features(
    image: NDArray, config: Dict[str, Any], is_panorama: bool
) -> Tuple[NDArray, NDArray, NDArray]:
    """Detect features in a color or gray-scale image.

    The type of feature detected is determined by the ``feature_type``
    config option.

    The coordinates of the detected points are returned in normalized
    image coordinates.

    Parameters:
        - image: a color image with shape (h, w, 3) or
                 gray-scale image with (h, w) or (h, w, 1)
        - config: the configuration structure
        - is_panorama : if True, alternate settings are used for feature count and extraction size.

    Returns:
        tuple:
        - points: ``x``, ``y``, ``size`` and ``angle`` for each feature
        - descriptors: the descriptor of each feature
        - colors: the color of the center of each feature
    """
    extraction_size = (
        config["feature_process_size_panorama"]
        if is_panorama
        else config["feature_process_size"]
    )
    features_count = (
        config["feature_min_frames_panorama"]
        if is_panorama
        else config["feature_min_frames"]
    )

    assert image.ndim == 2 or image.ndim == 3 and image.shape[2] in [1, 3]
    assert image.shape[0] > 2 and image.shape[1] > 2
    assert np.issubdtype(image.dtype, np.uint8)

    image = resized_image(image, extraction_size)
    if image.ndim == 2:  # convert (h, w) to (h, w, 1)
        image = np.expand_dims(image, axis=2)

    # convert color to gray-scale if necessary
    if image.shape[2] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image
    feature_type = config["feature_type"].upper()
    if feature_type == "SIFT":
        points, desc = extract_features_sift(image_gray, config, features_count)
    elif feature_type == "DDS":
        points, desc = extract_features_dds(image_gray, config, features_count)
    elif feature_type == "SURF":
        points, desc = extract_features_surf(image_gray, config, features_count)
    elif feature_type == "AKAZE":
        points, desc = extract_features_akaze(image_gray, config, features_count)
    elif feature_type == "HAHOG":
        points, desc = extract_features_hahog(image_gray, config, features_count)
    elif feature_type == "ORB":
        points, desc = extract_features_orb(image_gray, config, features_count)
    else:
        raise ValueError(
            "Unknown feature type (must be SURF, SIFT, AKAZE, HAHOG, ORB or DDS)"
        )

    xs = points[:, 0].round().astype(int)
    ys = points[:, 1].round().astype(int)
    # Clip por seguridad: si por algún motivo el modelo devuelve coords fuera
    # del frame, evitamos un IndexError en image[ys, xs].
    h_img, w_img = image.shape[:2]
    xs = np.clip(xs, 0, w_img - 1)
    ys = np.clip(ys, 0, h_img - 1)
    colors = image[ys, xs]
    if image.shape[2] == 1:
        colors = np.repeat(colors, 3).reshape((-1, 3))

    return normalize_features(points, desc, colors, image.shape[1], image.shape[0])


def build_flann_index(descriptors: NDArray, config: Dict[str, Any]) -> cv2.flann_Index:
    # FLANN_INDEX_LINEAR = 0
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_KMEANS = 2
    # FLANN_INDEX_COMPOSITE = 3
    # FLANN_INDEX_KDTREE_SINGLE = 4
    # FLANN_INDEX_HIERARCHICAL = 5
    FLANN_INDEX_LSH = 6

    if descriptors.dtype.type is np.float32:
        algorithm_type = config["flann_algorithm"].upper()
        if algorithm_type == "KMEANS":
            FLANN_INDEX_METHOD = FLANN_INDEX_KMEANS
        elif algorithm_type == "KDTREE":
            FLANN_INDEX_METHOD = FLANN_INDEX_KDTREE
        else:
            raise ValueError("Unknown flann algorithm type must be KMEANS, KDTREE")
        flann_params = {
            "algorithm": FLANN_INDEX_METHOD,
            "branching": config["flann_branching"],
            "iterations": config["flann_iterations"],
            "tree": config["flann_tree"],
        }
    elif descriptors.dtype.type is np.uint8:
        flann_params = {
            "algorithm": FLANN_INDEX_LSH,
            "table_number": 10,
            "key_size": 24,
            "multi_probe_level": 1,
        }
    else:
        raise ValueError(
            f"FLANN isn't supported for feature type {descriptors.dtype.type}."
        )

    return context.flann_Index(descriptors, flann_params)


def extract_features_surf(
    image: NDArray, config: Dict[str, Any], features_count: int
) -> Tuple[NDArray, NDArray]:
    surf_hessian_threshold = config["surf_hessian_threshold"]
    if context.OPENCV3:
        try:
            detector = cv2.xfeatures2d.SURF_create()
        except AttributeError as ae:
            if "no attribute 'xfeatures2d'" in str(ae):
                logger.error(
                    "OpenCV Contrib modules are required to extract SURF features"
                )
            raise
        descriptor = detector
        detector.setHessianThreshold(surf_hessian_threshold)
        detector.setNOctaves(config["surf_n_octaves"])
        detector.setNOctaveLayers(config["surf_n_octavelayers"])
        detector.setUpright(config["surf_upright"])
    else:
        detector = cv2.FeatureDetector_create("SURF")
        descriptor = cv2.DescriptorExtractor_create("SURF")
        detector.setDouble("hessianThreshold", surf_hessian_threshold)
        detector.setDouble("nOctaves", config["surf_n_octaves"])
        detector.setDouble("nOctaveLayers", config["surf_n_octavelayers"])
        detector.setInt("upright", config["surf_upright"])

    while True:
        logger.debug("Computing surf with threshold {0}".format(surf_hessian_threshold))
        t = time.time()
        if context.OPENCV3:
            detector.setHessianThreshold(surf_hessian_threshold)
        else:
            detector.setDouble(
                "hessianThreshold", surf_hessian_threshold
            )  # default: 0.04
        points = detector.detect(image)
        logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))
        if len(points) < features_count and surf_hessian_threshold > 0.0001:
            surf_hessian_threshold = (surf_hessian_threshold * 2) / 3
            logger.debug("reducing threshold")
        else:
            logger.debug("done")
            break

    points, desc = descriptor.compute(image, points)

    if desc is not None:
        if config["feature_root"]:
            desc = root_feature(desc)
        points = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
    else:
        points = np.array(np.zeros((0, 3)))
        desc = np.array(np.zeros((0, 3)))
    return points, desc
