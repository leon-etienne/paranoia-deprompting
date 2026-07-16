# Prompt reconstruction · Search without writing

A genetic algorithm mutates token IDs and keeps sequences that move closer to a
reference image in CLIP space.

![Prompt-reconstruction reference image](../../docs/images/prompt-reconstruction-source.jpg)

The presentation example uses **16 tokens, 200 iterations, and a population of
2,048**.

Source: [`picsum.photos/seed/42/1024/1024`](https://picsum.photos/seed/42/1024/1024)

<table>
  <tr>
    <th>Original</th>
    <th>Token prompt</th>
    <th>FLUX.1-dev</th>
    <th>SD 3.5 Medium</th>
    <th>SDXL Base 1.0</th>
    <th>SD v1.5</th>
  </tr>
  <tr>
    <td><img src="../../docs/images/prompt-reconstruction-source.jpg" width="140" alt="Original"></td>
    <td><code>email defeats&lt;/w&gt; wallart&lt;/w&gt; cave&lt;/w&gt; … mediter eleng</code></td>
    <td><img src="../../docs/images/reconstruction-flux1-dev.jpg" width="140" alt="FLUX"></td>
    <td><img src="../../docs/images/reconstruction-sd35-medium.jpg" width="140" alt="SD 3.5"></td>
    <td><img src="../../docs/images/reconstruction-sdxl-base.jpg" width="140" alt="SDXL"></td>
    <td><img src="../../docs/images/reconstruction-sd15.jpg" width="140" alt="SD 1.5"></td>
  </tr>
</table>

```bash
python scripts/evolution/image_to_tokens.py
python scripts/evolution/tokens_to_image.py
python scripts/evolution/app.py
```

- Search is guided by CLIP similarity.
- Rendering uses SDXL Turbo.
- CLIP and SDXL vocabularies are related, not interchangeable.
- CUDA and substantial VRAM are expected.

[Notebook](../../notebooks/3_Evolution.ipynb)
