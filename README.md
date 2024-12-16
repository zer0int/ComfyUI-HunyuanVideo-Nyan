# ComfyUI-HunyuanVideo-Nyan
Text Encoders finally matter 🤖🎥 - scale CLIP &amp; LLM influence! + a Nerdy Transformer Shuffle node
----
- The CLIP model doesn't seem to matter much? True for default Hunyuan Video, False with this node! ✨
- Simply put the `ComfyUI...` folder from this repo in `ComfyUI/custom_nodes`
- See example workflow; it's really easy to use, though. Replaces the loader node.
- Recommended CLIP [huggingface.co/zer0int/CLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14)
- Takes 248 tokens, a bit unpredictable: [huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14)
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

https://github.com/user-attachments/assets/ff234efa-af12-4abf-9a1d-1563032d789e

https://github.com/user-attachments/assets/a50d7b71-7325-4dfa-948a-3eb237a4d425

- Q: And what does this confusing, gigantic node for nerds do? 🤓 
- A: You can glitch the transformer (video model) by shuffling or skipping MLP and Attention layers:

https://github.com/user-attachments/assets/72f9746a-77c7-4710-90ac-15516d04fc73

