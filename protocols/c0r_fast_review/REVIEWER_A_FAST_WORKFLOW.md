# Reviewer A fast local workflow

This workflow reviews whether each displayed image, question, and target/reference are mutually valid. It does not evaluate model outputs or editing methods.

## Shortcuts

- `1`: direct valid answer
- `2`: valid image-dependent or spatial reference
- `3`: context-dependent but acceptable
- `4`: clearly invalid; choose an issue and enter a reason of at least 12 characters
- `5`: unresolved; choose an ambiguity issue and enter a reason of at least 12 characters
- `6`: custom verdict with no defaults
- `Enter`: submit the selected and validated verdict; presets 4 and 5 require a second confirmation
- `Esc`: clear the unsubmitted selection
- `F`: move the item to the end of the local queue without creating a verdict
- `+`, `-`, `0`: browser-only image zoom controls

No verdict is selected on page load. Every item requires an explicit human choice. Do not upload, copy, crop, screenshot, transcode, cache, or redistribute any source image.

After 200/200 items are complete, stop the local server and run `reviewctl.py freeze-reviewer-a`. Reviewer B preparation is a separate command and does not start a review server.
