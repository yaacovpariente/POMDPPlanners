# CartPole artwork

Original lab and cart images were generated with the built-in image generation tool for this renderer. The lab prompt requested a softly lit neutral workshop backdrop with no stateful objects, rail or text. The cart prompt requested a compact blue enamel laboratory trolley with silver trim, two small wheels and a centered bearing, on true transparent alpha with no pole or rail.

Runtime copies are 800×500 RGB (`lab.png`) and 192×120 RGBA (`cart.png`). The cart is framed to the physical object's alpha bounds. Small packaged copies and bounded runtime caches avoid decoding full-size generated art on every saved episode. Rendering never calls an image service.

Only material appearance comes from these files. Cart position, horizontal rail, pole endpoints, thresholds, force direction and all labels come from recorded data and environment parameters.
