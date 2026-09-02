# M3Bench editor one-edit effect gate V2

All four editor implementations passed the real-model one-edit effect contract on GPU2 in independent fresh processes. Every method used a real image tensor, expanded multimodal inputs, a nonempty target-only mask, a finite positive update, lower post-edit target NLL, nonempty generation, save/reload parity, reset/disable base parity, and unchanged frozen base weights. Routed methods also passed self-route, stable generation route lifetime, and miss/base parity checks.

This is a development gate on one approved smoke record, not a formal benchmark result.
