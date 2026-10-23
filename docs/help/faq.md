# FAQ

??? question "Do I need a breathing sensor?"
    Only to train and to score. Predicting on new videos needs only the video and its
    frame timestamps. To train a network for your rig, you need some recordings with a
    sensor (a nasal thermistor, or any sensor whose trace follows breathing) beside the
    face camera.

??? question "Can I use a network trained on someone else's rig?"
    Yes, if your `[preprocess]` recipe matches the one it trained with: frame size, box
    size, `select_fs` and `output_fs`. `clips scan --checkpoint` writes a list with the
    checkpoint's recipe, and the app's Predict → Videos places crops at its size. A
    different camera angle or lighting can lower its accuracy: judge it with [quality
    control](../guides/quality-control.md), or fine-tune it on a few of your own
    labelled sessions (`init_from`).

??? question "Does zephyr use colour, optical flow or several crops?"
    No. Gray frames from one crop box are the only input. The network learns motion from
    the sequence of frames.

??? question "My camera runs at 30 Hz. Can I use it?"
    Yes. Set `select_fs = 30` (or lower) in `[preprocess]`. The prediction is still made
    and scored on a 60 Hz grid. Event tolerance is 1.5 frames of the output grid, so it
    widens with a slower `output_fs`.

??? question "Why are inhales read from the onset head, not from peaks on the trace?"
    The onset head is trained to find inhales, and finds them more reliably than a peak
    detector run on the predicted trace. Peaks on the trace (DSP) are still used for
    exhales and for networks without an onset head, and are reported as `inhale_f1`
    beside the head's `head_inhale_f1`.

??? question "Why hold out whole animals?"
    Sessions of one animal share its face, its fur and its rig position. A network
    scored on a new session of an animal it trained on looks better than it will on a
    new animal. Hold out whole animals for testing and validation, and rotate which
    ones.

??? question "How long are clips? Is there a length limit?"
    There is no limit. Inference runs in overlapping windows and stitches them, so a
    long clip gives the same trace as one unbounded pass. The tutorial's clips are about
    5 minutes long.

??? question "Can I run zephyr on a cluster?"
    Yes. Everything but placing boxes and choosing detectors runs headless. Install
    without the `ui` extra, and use the commands the app shows under **Copy command**.
    See [Working across machines](../guides/machines.md).

??? question "Where do I place the crop box?"
    Over the nostrils and mouth, with some cheek and whisker pad around them, the same
    size for every clip. Keep the eye out if you can. See [Place crop
    boxes](../tutorial/boxes.md).

??? question "What does each score mean?"
    See [Scoring](../reference/scoring.md).
