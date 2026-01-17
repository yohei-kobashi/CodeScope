# miyabiにおけるCodeScopeの評価実行
## 1. 推論時の環境構築について
推論ではminicondaからvllmをインストールして実行しています。具体的には松尾岩澤研のnotionを参考にして下さい。  
以下は実行時のmoduleと環境変数の設定および構築済みの環境の情報です。
```
(inference_env) [b20048@miyabi-g2 ~]$ qsub -I -l select=1 -W group_list=gj26 -q interact-g
qsub: waiting for job 1268422.opbs to start
qsub: job 1268422.opbs ready

[b20048@mg0013 ~]$ module purge
module load cuda/12.8
module load cudnn/9.10.1.4
module load nvidia/25.3
module load nv-hpcx/25.3
source /work/gj26/b20048/miniconda3/etc/profile.d/conda.sh
conda activate inference_env
export CUDA_VISIBLE_DEVICES=0
export PATH="$CONDA_PREFIX/bin:/opt/rh/gcc-toolset-14/root/usr/bin:$PATH"

export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
export TRITON_CC="$CC"
export TRITON_CXX="$CXX"
export CUDAHOSTCXX="$CXX"

export PYTHONNOUSERSITE=1
(inference_env) [b20048@mg0013 ~]$ conda --version
conda 25.3.1
(inference_env) [b20048@mg0013 ~]$ module list
Currently Loaded Modulefiles:
 1) cuda/12.8   2) cudnn/9.10.1.4   3) nvidia/25.3   4) nv-hpcx/25.3  
(inference_env) [b20048@mg0013 ~]$ pip freeze
aiohappyeyeballs==2.6.1
aiohttp==3.12.15
aiosignal==1.4.0
airportsdata==20250909
annotated-types==0.7.0
anyio==4.10.0
astor==0.8.1
asttokens==3.0.0
attrs==25.3.0
blake3==1.0.5
build==1.3.0
cachetools==6.2.0
certifi==2025.8.3
charset-normalizer==3.4.3
click==8.2.1
cloudpickle==3.1.1
cmake==4.1.0
compressed-tensors==0.9.4
cupy-cuda12x==13.6.0
datasets==4.1.0
decorator==5.2.1
depyf==0.18.0
dill==0.4.0
diskcache==5.6.3
distro==1.9.0
dnspython==2.8.0
einops==0.8.1
email-validator==2.3.0
executing==2.2.1
fastapi==0.116.2
fastapi-cli==0.0.11
fastapi-cloud-cli==0.1.5
fastrlock==0.8.3
filelock==3.19.1
frozenlist==1.7.0
fsspec==2024.6.1
gguf==0.17.1
googleapis-common-protos==1.70.0
grpcio==1.75.0
h11==0.16.0
hf-xet==1.1.10
httpcore==1.0.9
httptools==0.6.4
httpx==0.28.1
huggingface-hub==0.35.0
idna==3.10
importlib_metadata==8.7.0
interegular==0.3.3
ipython==9.5.0
ipython_pygments_lexers==1.1.1
jedi==0.19.2
Jinja2==3.1.6
jiter==0.11.0
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
lark==1.2.2
lit==18.1.8
llguidance==0.7.30
llvmlite==0.44.0
lm-format-enforcer==0.10.12
markdown-it-py==4.0.0
MarkupSafe==2.1.5
matplotlib-inline==0.1.7
mdurl==0.1.2
mistral_common==1.8.5
mpmath==1.3.0
msgpack==1.1.1
msgspec==0.19.0
multidict==6.6.4
multiprocess==0.70.16
nest-asyncio==1.6.0
networkx==3.3
ninja==1.13.0
numba==0.61.2
numpy==2.1.2
openai==1.107.3
opencv-python-headless==4.12.0.88
opentelemetry-api==1.37.0
opentelemetry-exporter-otlp==1.37.0
opentelemetry-exporter-otlp-proto-common==1.37.0
opentelemetry-exporter-otlp-proto-grpc==1.37.0
opentelemetry-exporter-otlp-proto-http==1.37.0
opentelemetry-proto==1.37.0
opentelemetry-sdk==1.37.0
opentelemetry-semantic-conventions==0.58b0
opentelemetry-semantic-conventions-ai==0.4.13
outlines==0.1.11
outlines_core==0.1.26
packaging==25.0
pandas==2.3.2
parso==0.8.5
partial-json-parser==0.2.1.1.post6
pexpect==4.9.0
pillow==11.0.0
prometheus-fastapi-instrumentator==7.1.0
prometheus_client==0.22.1
prompt_toolkit==3.0.52
propcache==0.3.2
protobuf==6.32.1
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.3
py-cpuinfo==9.0.0
pyarrow==21.0.0
pybind11==3.0.1
pycountry==24.6.1
pydantic==2.11.9
pydantic-extra-types==2.10.5
pydantic_core==2.33.2
Pygments==2.19.2
pyproject_hooks==1.2.0
python-dateutil==2.9.0.post0
python-dotenv==1.1.1
python-json-logger==3.3.0
python-multipart==0.0.20
pytz==2025.2
PyYAML==6.0.2
pyzmq==27.1.0
ray==2.49.1
referencing==0.36.2
regex==2025.9.1
requests==2.32.5
rich==14.1.0
rich-toolkit==0.15.1
rignore==0.6.4
rpds-py==0.27.1
safetensors==0.6.2
scipy==1.16.2
sentencepiece==0.2.1
sentry-sdk==2.38.0
setuptools==78.1.1
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
stack-data==0.6.3
starlette==0.48.0
sympy==1.13.3
tiktoken==0.11.0
tokenizers==0.21.4
torch==2.7.0+cu128
torchaudio==2.7.0
torchvision==0.22.0
tqdm==4.67.1
traitlets==5.14.3
transformers==4.52.4
triton @ file:///work/gj26/b20048/triton/python
typer==0.17.4
typing-inspection==0.4.1
typing_extensions==4.12.2
tzdata==2025.2
urllib3==2.5.0
uvicorn==0.35.0
uvloop==0.21.0
vllm @ file:///work/go25/share/swift_wheels/vllm-0.9.1.dev0%2Bg587387724.d20250613-cp312-cp312-linux_aarch64.whl#sha256=c70f5f413748eaaf0011335c09c729c161e168200f0c59545fc08e643846904c
watchfiles==1.1.0
wcwidth==0.2.13
websockets==15.0.1
wheel==0.45.1
xgrammar==0.1.19
xxhash==3.5.0
yarl==1.20.1
zipp==3.23.0
```
## 2. 推論
./run_vllm.shを参考に./inference/run_vllm.pyを実行して下さい。  
実行時のタスクとなる../../data/code_translation_data.jsonlはCodeScopeオリジナルのタスクを./convert_data.pyで編集したものです。
推論結果はデフォルトだと./resultディレクトリに保存されます。

## 3. 評価
./run_eval.shを参考に./evaluator/run_multiple.pyを実行してください。  
```
python evaluator/run_multiple.py --jsonl_path ./result/test.jsonl  --output_path ./result/scores_test.json
```
AWS lambda上の各言語評価関数にリクエストを出し、戻ってきた結果を集計します。  