# ComfyUI-HunyuanVideo-Nyan

### Text Encoders finally matter 🤖🎥 - scale CLIP &amp; LLM influence! 
+ plus, a Nerdy Transformer Shuffle node
----
## Changes 28/DEC/24:

- There are many ways to use an LLM for prompting, but just one (afaik) that lets you mess with the LLM's attention:
- [https://github.com/zer0int/ComfyUI-LLama3-Layer-Shuffle-Prompting](https://github.com/zer0int/ComfyUI-LLama3-Layer-Shuffle-Prompting)
- Shuffle the AI's attention & cause a confusion! Embrace the completely unpredictable!
- Workflow included in the 24-DEC workflows folder. Just install the above node first.


https://github.com/user-attachments/assets/38a9af77-de4c-4838-a4fc-9f6b0c0ce5b2


----
## Changes 🎄 24/DEC/2024:
- Fix node for compatibility with [kijai/ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper)
- Now with Timestamp to log compatibility (use same as previous version, see below)
- Include updated Image-To-Video + Text-To-Video workflows
----
## Changes 19/DEC/2024:
- New (best) SAE-informed Long-CLIP model with 90% ImageNet/ObjectNet accuracy.
- Code is here, model is at my HF 🤗: [https://huggingface.co/zer0int/LongCLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-SAE-ViT-L-14)
----
- To clarify, only put this folder into `ComfyUI/custom_nodes`; if you cloned the entire repo, you'll need to move it. `only this!` should be in `ComfyUI/custom_nodes`; you should have an `__init__.py` in your `ComfyUI/custom_nodes/ComfyUI-HunyuanVideo-Nyan` folder. If you see a README.md, that's wrong.

![clarify](https://github.com/user-attachments/assets/a8d9a977-eb31-4bc5-b187-18d96e5bfe6c)
----
- The CLIP model doesn't seem to matter much? True for default Hunyuan Video, False with this node! ✨
- Simply put the `ComfyUI...` folder from this repo in `ComfyUI/custom_nodes`
- See example workflow; it's really easy to use, though. Replaces the loader node.
- Recommended CLIP [huggingface.co/zer0int/CLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14)
- Takes 248 tokens, 🆕 @ 19/DEC/24 🤗: [https://huggingface.co/zer0int/LongCLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-SAE-ViT-L-14)

![use-node](https://github.com/user-attachments/assets/59928c01-3118-4be4-b31c-037b32073f26)

- Requires [kijai/ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper)
- ⚠️ If something breaks because WIP: Temporarily fall back to [my fork](https://github.com/zer0int/ComfyUI-HunyuanVideoWrapper) for compatibility
- Uses HunyuanVideoWrapper -> loader node implementation. All credits to the original author!
- My code = only the 2 different 'Nyan nodes' in `hynyan.py`.
- Loader is necessary as the mod changes model buffers; changes are cumulative if not re-loaded.
- You can choose to re-load from file - or from RAM deepcopy (faster, *may* require >64 GB RAM).

![two-nodes](https://github.com/user-attachments/assets/7dfe165f-ab03-4c52-bad6-2a1410c5bf3d)

- Q: What does it do, this `Factor` for scaling CLIP & LLM? 🤔 
- A: Here are some examples. Including a 'do NOT set BOTH the CLIP and LLM factors >1' example.
- Prompt: `high quality nature video of a red panda balancing on a bamboo stick while a bird lands on the panda's head, there's a waterfall in the background`
- SAE: Bird at least flies (though takes off), better feet on panda (vs. OpenAI)

https://github.com/user-attachments/assets/ff234efa-af12-4abf-9a1d-1563032d789e

- These are all my CLIP models from [huggingface.co/zer0int](https://huggingface.co/zer0int); SAE is best.
- See details on legs; blurriness; coherence of small details.

https://github.com/user-attachments/assets/a50d7b71-7325-4dfa-948a-3eb237a4d425

----
🆕 Long-CLIP @ 19/DEC/24:
The original CLIP model has 77 tokens max input - but only ~20 tokens effective length. See the [original Long-CLIP paper](https://arxiv.org/abs/2403.15378) for details. HunyuanVideo demo:
- 69 tokens, normal scene:
1. Lens: 16mm. Aperture: f/2.8. Color Grading: Blue-green monochrome. Lighting: Low-key with backlit silhouettes. Background: Gothic cathedral at night, stained glass windows breaking. Camera angle: Over the shoulder of a ninja, tracking her mid-air leap as she lands on a rooftop.
- 52 tokens, OOD (Out-of-Distribution) scene: Superior handling for consistency and prompt-following despite OOD concept.
2. In this surreal nightmare documentary, a sizable spider with a human face is peacefully savoring her breakfast at a diner. The spider has a spider body, but a lady's face on the front, and regular human hands at the end of the spider legs.



https://github.com/user-attachments/assets/d424e089-1243-4510-9561-61c8ad5ea5b0


----

- Q: And what does this confusing, gigantic node for nerds do? 🤓 
- A: You can glitch the transformer (video model) by shuffling or skipping MLP and Attention layers:

https://github.com/user-attachments/assets/72f9746a-77c7-4710-90ac-15516d04fc73

