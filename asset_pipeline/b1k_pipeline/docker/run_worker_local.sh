#!/bin/bash

# VALUES TO SET #######################################
DOCKER_IMAGE="stanfordvl/ig_pipeline:latest"     # Can also use, e.g.: stanfordvl/omnigibson:latest
SCR_DIR="/scr"
CVGL2_DIR="/cvgl2"
#######################################################
# YOU SHOULD NOT HAVE TO TOUCH ANYTHING BELOW HERE :) #


BYellow='\033[1;33m'
Color_Off='\033[0m'

# Parse the command line arguments.
ENV_KWARGS=""
case $1 in
    -n|--no-omniverse)
    ENV_KWARGS="${ENV_KWARGS} --env OMNIGIBSON_NO_OMNIVERSE=1"
    shift
    ;;
esac


SCRIPT_DIR="/scr/ig_pipeline/b1k_pipeline/docker"
DATA_PATH="${SCRIPT_DIR}/data"
ISAAC_CACHE_PATH="${SCR_DIR}/isaac_cache"

ICD_PATH_1="/usr/share/vulkan/icd.d/nvidia_icd.json"
ICD_PATH_2="/etc/vulkan/icd.d/nvidia_icd.json"
LAYERS_PATH_1="/usr/share/vulkan/icd.d/nvidia_layers.json"
LAYERS_PATH_2="/usr/share/vulkan/implicit_layer.d/nvidia_layers.json"
LAYERS_PATH_3="/etc/vulkan/implicit_layer.d/nvidia_layers.json"
EGL_VENDOR_PATH="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

# Find the ICD file
if [ -e "$ICD_PATH_1" ]; then
    ICD_PATH=$ICD_PATH_1
elif [ -e "$ICD_PATH_2" ]; then
    ICD_PATH=$ICD_PATH_2
else
    echo "Missing nvidia_icd.json file.";
    echo "Typical paths:";
    echo "- /usr/share/vulkan/icd.d/nvidia_icd.json or";
    echo "- /etc/vulkan/icd.d/nvidia_icd.json";
    echo "You can google nvidia_icd.json for your distro to find the correct path.";
    echo "Consider updating your driver to 525 if you cannot find the file.";
    echo "To continue update the ICD_PATH_1 at the top of the run_docker.sh file and retry";
    exit;
fi

# Find the layers file
if [ -e "$LAYERS_PATH_1" ]; then
    LAYERS_PATH=$LAYERS_PATH_1
elif [ -e "$LAYERS_PATH_2" ]; then
    LAYERS_PATH=$LAYERS_PATH_2
elif [ -e "$LAYERS_PATH_3" ]; then
    LAYERS_PATH=$LAYERS_PATH_3
else
    echo "Missing nvidia_layers.json file."
    echo "Typical paths:";
    echo "- /usr/share/vulkan/icd.d/nvidia_layers.json";
    echo "- /usr/share/vulkan/implicit_layer.d/nvidia_layers.json";
    echo "- /etc/vulkan/implicit_layer.d/nvidia_layers.json";
    echo "You can google nvidia_layers.json for your distro to find the correct path.";
    echo "Consider updating your driver to 525 if you cannot find the file.";
    echo "To continue update the LAYERS_PATH_1 at the top of the run_docker.sh file and retry";
    exit;
fi

if [ ! -e "$EGL_VENDOR_PATH" ]; then
    echo "Missing ${EGL_VENDOR_PATH} file."
    echo "(default path: /usr/share/vulkan/icd.d/nvidia_icd.json)";
    echo "To continue update the EGL_VENDOR_PATH at the top of the run_docker.sh file and retry";
    exit;
fi

# Define env kwargs to pass
declare -A ENVS=(
    [NVIDIA_DRIVER_CAPABILITIES]=all
    [DISPLAY]=""
    [OMNIGIBSON_HEADLESS]=1
)
for env_var in "${!ENVS[@]}"; do
    # Add to env kwargs we'll pass to enroot command later
    ENV_KWARGS="${ENV_KWARGS} --env ${env_var}=${ENVS[${env_var}]}"
done

# Define mounts to create (maps local directory to container directory)
declare -A MOUNTS=(
    [${SCR_DIR}]=/scr
    [${DATA_PATH}]=/data
    [${ICD_PATH}]=/etc/vulkan/icd.d/nvidia_icd.json
    [${LAYERS_PATH}]=/etc/vulkan/implicit_layer.d/nvidia_layers.json
    [${EGL_VENDOR_PATH}]=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
    [${ISAAC_CACHE_PATH}/kit/cache/Kit]=/isaac-sim/kit/cache/Kit
    [${ISAAC_CACHE_PATH}/cache/ov]=/root/.cache/ov
    [${ISAAC_CACHE_PATH}/cache/pip]=/root/.cache/pip
    [${ISAAC_CACHE_PATH}/cache/glcache]=/root/.cache/nvidia/GLCache
    [${ISAAC_CACHE_PATH}/cache/computecache]=/root/.nv/ComputeCache
    [${ISAAC_CACHE_PATH}/logs]=/root/.nvidia-omniverse/logs
    [${ISAAC_CACHE_PATH}/config]=/root/.nvidia-omniverse/config
    [${ISAAC_CACHE_PATH}/data]=/root/.local/share/ov/data
    [${ISAAC_CACHE_PATH}/documents]=/root/Documents
)

MOUNT_KWARGS=""
for mount in "${!MOUNTS[@]}"; do
    # Verify mount path in local directory exists, otherwise, create it
    if [ ! -e "$mount" ]; then

        mkdir -p ${mount}
    fi
    # Add to mount kwargs we'll pass to enroot command later
    MOUNT_KWARGS="${MOUNT_KWARGS} --mount ${mount}:${MOUNTS[${mount}]}"
done

echo -e "${BYellow}IMPORTANT: Referencing OmniGibson assets at ${DATA_PATH}. ${Color_Off}"

#echo "The NVIDIA Omniverse License Agreement (EULA) must be accepted before"
#echo "Omniverse Kit can start. The license terms for this product can be viewed at"
#echo "https://docs.omniverse.nvidia.com/app_isaacsim/common/NVIDIA_Omniverse_License_Agreement.html"
#
#while true; do
#    read -p "Do you accept the Omniverse EULA? [y/n] " yn
#    case $yn in
#        [Yy]* ) break;;
#        [Nn]* ) exit;;
#        * ) echo "Please answer yes or no.";;
#    esac
#done

# Only run this if there is no sqsh file
SQSH_SOURCE="${SCRIPT_DIR}/ig_pipeline.sqsh"
if [ ! -e ${SQSH_SOURCE} ]; then
    echo "Could not find valid sqsh file at ${SQSH_SOURCE}, creating..."
    enroot import -o $SQSH_SOURCE "docker://${DOCKER_IMAGE}"
fi

# Remove leading space in string
ENV_KWARGS="${ENV_KWARGS:1}"
MOUNT_KWARGS="${MOUNT_KWARGS:1}"


# Create the image if it doesn't already exist
WORKER_CNT=$1; shift
for ((i = 1 ; i <= $WORKER_CNT ; i++));
do
    CONTAINER_NAME=ig_pipeline_${i}
    echo "Creating container ${CONTAINER_NAME}..."
    # enroot create --force --name ${CONTAINER_NAME} ${SQSH_SOURCE}

    if [ `expr $i % 2` == 0 ]
    then
        GPU=0
    else
        GPU=1
    fi

    echo "Launching job"
    export ENROOT_RESTRICT_DEV=y
    enroot start \
        --root \
        --rw \
        ${ENV_KWARGS} \
        --env OMNIGIBSON_GPU=${GPU} \
        ${MOUNT_KWARGS} \
        ${CONTAINER_NAME} $@ &> /dev/null &
done

# Uncomment this to remove the image after
# for i in {1..8}
# do
#     enroot remove -f ig_pipeline_${i}
# done

