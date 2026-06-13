# AI Developer Hand-off: Autonomous Geometry Dash RL Agent

## 1. Project Context & Objectives

The goal of this project is to build a Reinforcement Learning (RL) agent capable of playing Geometry Dash (specifically the deterministic level "Stereo Madness") in real-time. The game is running via the BlueStacks Android emulator on Windows.

The project operates strictly from raw pixel inputs (no internal game state APIs) and emulates physical keystrokes.

**Development Philosophy:**
The codebase must be treated as a systems-engineering project. Memory management (avoiding leaks during rapid frame capture) and processing latency are critical bottlenecks. The architecture is split into two distinct phases to ensure a rapid prototype to production pipeline.

## 2. The Two-Phase Architecture

### Phase 1: Rapid Prototyping & Systems Integration (Current Focus)

Focuses on building the computer vision pipeline, standardizing the environment, and utilizing off-the-shelf algorithms to achieve a working prototype.

* **The Brain:** `stable-baselines3` (DQN algorithm).
* **The Environment:** Custom wrapper inheriting from `gymnasium.Env`.
* **The Eyes:** `mss` (screen capture) and `opencv-python` (image processing).
* **The Hands:** `pydirectinput` (DirectX-compatible keystrokes).

### Phase 2: Algorithmic Deep Dive (Future Scope)

Once Phase 1 successfully clears the level, `stable-baselines3` will be stripped out. The Gymnasium environment and vision pipeline will remain.

* **The Brain:** A custom Convolutional Neural Network (CNN), Replay Memory Buffer, and Deep Q-Learning training loop written entirely from scratch in `PyTorch`.

---

## 3. Core System Loop (30+ FPS Target)

The system must continuously execute the following loop with minimal latency:

1. **Capture:** Extract the specific BlueStacks bounding box using `mss`.
2. **Pre-process:** Convert the frame to grayscale, resize to 84x84, and apply thresholding via OpenCV.
3. **State Representation:** Stack the current frame with the previous 3 frames (`shape=(4, 84, 84)`) to capture velocity and momentum.
4. **Decision:** Pass the stacked frames to the DQN to output an action (0: Do Nothing, 1: Jump).
5. **Actuation:** If action is 1, trigger `pydirectinput.press('space')`.
6. **Observation & Reward:** Calculate the reward (+1 for surviving the frame, -100 for death detected via screen flash/text pixel triggers).

---

## 4. Implementation Steps & File Structure

The AI assistant is expected to help generate and refine the following modules:

### A. `vision.py` (Perception Module)

* **Responsibility:** Ultra-fast screen capture and image processing.
* **Expectations:** Must not use standard `PIL.ImageGrab` or standard `pyautogui` screenshots due to high latency. Must include functions to isolate the game area and detect "Death" conditions using template matching (`cv2.matchTemplate`) or strict pixel color bounds.

### B. `controls.py` (Actuation Module)

* **Responsibility:** Emulating keyboard input.
* **Expectations:** Must use `pydirectinput` to bypass emulator input blocks. Keep latency to an absolute minimum.

### C. `env.py` (The Gymnasium Wrapper)

* **Responsibility:** Bridging the raw game data with standard RL APIs.
* **Expectations:**
* Inherit from `gymnasium.Env`.
* Define `action_space` as `Discrete(2)`.
* Define `observation_space` as `Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)`.
* Implement robust `reset()` and `step()` functions. `reset()` must handle clicking the in-game restart button.

### D. `train.py` (Phase 1 Execution)

* **Responsibility:** Initializing Stable Baselines3 and managing the training loop.
* **Expectations:** Wrap the environment in `VecFrameStack`, configure the `DQN` with a `CnnPolicy`, establish TensorBoard logging, and manage model checkpointing to save progress every `X` timesteps.

---

## 5. Strict Coding Directives for the AI Assistant

When generating code or suggesting refactors, the AI assistant must adhere to the following rules:

* **Latency is the Enemy:** Do not introduce heavy library calls inside the main `step()` loop. OpenCV operations must be heavily optimized.
* **Explicit Tensor/Array Shapes:** When dealing with image arrays and PyTorch tensors, always document the expected shape transformations in inline comments (e.g., `(Height, Width, Channels) -> (Channels, Height, Width)`).
* **Memory Discipline:** Ensure image arrays are properly overwritten or garbage collected. Storing raw uncompressed frames in the replay buffer will cause an Out-Of-Memory (OOM) crash.
* **Modular Design:** The CV pipeline and the RL environment must remain decoupled so they can be easily reused in Phase 2 without rewriting the vision logic.
* **Logging:** Include standard Python `logging` for critical state changes (e.g., "Game Over Detected", "Model Checkpoint Saved").
