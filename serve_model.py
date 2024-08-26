from internvl_inference import ImageInferenceModel
from ray import serve
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

# Define a request model
class InferenceRequest(BaseModel):
    image: str  # Assuming image is sent as a base64-encoded string
    prompt: str

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 1})
@serve.ingress(app)
class ImageInferenceDeployment:
    def __init__(self, model_path, device='cuda', input_size=448, max_num=12):
        self.model = ImageInferenceModel(model_path, device, input_size, max_num)

    @app.post("/infer")
    async def infer(self, request: InferenceRequest):
        image = request.image
        prompt = request.prompt
        response = self.model.infer(image, prompt)
        return {"response": response}


app = ImageInferenceDeployment.bind("OpenGVLab/InternVL2-8B")
