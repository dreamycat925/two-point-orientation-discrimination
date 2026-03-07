# Two-Point Orientation Discrimination

A Streamlit-based examiner-assist app for **two-point orientation discrimination testing** with **JVP domes**.

## Live App

Use the public app here:

**https://two-point-orientation-discrimination-dwdjixy5ixuug3mbh4jgvf.streamlit.app/**

## Overview

This app is designed to support examiner-operated testing for **grating orientation discrimination** using JVP domes.

The app displays:
- the **next dome size** to use,
- the **next orientation** to present (**vertical** or **horizontal**),
- response buttons for the examiner to record the participant's answer,
- and a running summary of the current phase, trial count, reversals, and provisional threshold.

The app follows a simple workflow:
1. **Practice**
2. **Main test**
3. **Post-test check**

## Dome Sizes

The app uses the following dome sizes:

```text
0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 12.0 mm
```

**Blank domes are not used.**

## Practice Phase

- Starts at **8 mm**
- Orientation is **vertical/horizontal with 50/50 probability**
- Ends with **PASS** after **5 consecutive correct responses**
- If the participant makes **2 errors at 8 mm**, the app steps up to **10 mm**
- If the participant makes **2 errors at 10 mm**, the app steps up to **12 mm**
- If the participant makes **2 errors at 12 mm**, the phase ends as **FAIL**

## Main Test Phase

- Starts at **8 mm**
- Uses **2-down 1-up** staircase logic
- Dome size changes by **one adjacent step at a time**
- Orientation schedule can be selected from:
  - **Series 1**
  - **Series 2**
  - **Random**
- Default schedule: **Series 1**

### Main test stopping rules

The main test ends when any of the following occurs:

- **PASS**: **4 consecutive correct responses at 0.35 mm**
- **FAIL**: **2 consecutive incorrect responses at 12.0 mm**
- **Converged**: **10 reversals** reached
- **Non-convergent**: **100 trials** reached without meeting another stopping rule

### Provisional threshold

The app displays a provisional threshold as:

- **median of the last 6 reversals**

## Post-Test Check

The post-test check uses the **same rules as the practice phase**.

This gives a simple indirect check of whether task engagement and performance remained stable through the session.

## Output

The app can export:
- a **CSV log** containing all phases,
- and the **orientation schedule** used in the main test.

## Quick Use

1. Open the live app.
2. Run the **practice phase**.
3. Start the **main test**.
4. Run the **post-test check**.
5. Download the **CSV log** if needed.

## Local / Docker

Most users can use the public Streamlit app above.

For local execution:

```bash
pip install -r requirements.txt
streamlit run two_point_orientation_discrimination_streamlit_app.py
```

For Docker-based local use, see **README_DOCKER.md**.

## Notes

- This app is intended as an **examiner support tool**.
- It is not a medical device.
- Testing conditions, participant attention, and examiner technique can affect results.
