This is a very rough port of the diffusers instruct pix2pix pipeline to ONNX. You should be able to just convert a model with the same scripts without modification as long as you use the correct yaml file. Get the model here: https://huggingface.co/timbrooks/instruct-pix2pix
<br>To test it, just edit line 6 of testip2p.py to point to where your converted ONNX model is and run it. There's no UI support yet, as it needs some more modifications, but it's at least at the "it works" stage.
