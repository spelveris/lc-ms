# LC-MS Analysis

Analyze LC-MS (Liquid Chromatography-Mass Spectrometry) data from `.D` folders.

## Group

This project is from the ETH Zürich group listed here:
[https://lang.ethz.ch/](https://lang.ethz.ch/)

## Features

- **UV-Vis Chromatograms**: Multi-wavelength detector (MWD) data at 194, 254, 280 nm
- **Mass Spectrometry**: Total Ion Chromatogram (TIC) and Extracted Ion Chromatograms (EIC)
- **Protein Deconvolution**: Local LC-MS machine-like neutral-mass deconvolution from charge envelopes
- **Time Progression**: Compare 2+ samples with color-coded overlays
- **Time Change MS**: Overlay summed mass spectra across 2+ samples
- **Batch Deconvolution**: Run default deconvolution across multiple samples
- **Mass Calculator**: Compare calculated and observed masses, annotate modifications
- **Report Export**: Generate multi-page PDF reports
- **Export**: Download plots as PNG (300 DPI), SVG, or PDF

## Download Desktop App

Download the app for your platform - no installation required:

| Platform | Download |
|----------|----------|
| **macOS** | [Latest Release Assets](../../releases/latest) |
| **Windows** | [Latest Release Assets](../../releases/latest) |

Release ZIP names include the version, for example:
- `LCMS-Analysis-v1.8.1-macOS.zip`
- `LCMS-Analysis-v1.8.1-Windows.zip`

### Installation

1. Download and extract the ZIP for your platform
2. **macOS**: Open Terminal, navigate to the folder, run `./LCMS-Analysis`
3. **Windows**: Double-click `LCMS-Analysis.exe`
4. Your browser will open with the app
5. Use **Browse Files** mode to select .D folders directly (no ZIP needed!)

## Web Version

Use the online version (requires ZIP upload): [Open Web App](https://lcms-lang.streamlit.app/)

## Run from Source

```bash
git clone https://github.com/spelveris/lc-ms.git
cd lc-ms
pip install -r requirements.txt
streamlit run app/main.py
```

## Configuration

### Environment Variables

- `LCMS_BASE_PATH`: Root directory for LC-MS data (default: `/data/lcms`)

### Default Settings

- UV Wavelengths: multi-select; 194 nm pre-selected when available
- UV Smoothing: 36
- EIC Smoothing: 5
- m/z Window: ±0.5
- Default m/z values: none (users add targets in the sidebar)
- Export DPI: 300
- Deconvolution method: Local LC-MS machine-like (default mode)
- Spectrum extraction: Average (sum)
- Show top N masses: 5
- Deconvolution MW range: 500 to 50000 Da
- Deconvolution charge range: 1 to 50
- Deconvolution noise cutoff: 1000 counts
- Min contiguous charges: 3
- Include singly-charged ions (z=1): off by default

## Usage

1. **Select Files**: Use the sidebar file browser to navigate to your LC-MS data
2. **Choose Analysis Mode**:
   - Single sample: View UV, TIC, and EICs for one file
   - Time progression: Compare 2+ samples across timepoints
   - EIC batch: Extract multiple EICs with peak areas
3. **Adjust Settings**: Modify UV wavelength, smoothing, and m/z window in sidebar
4. **Configure m/z Targets**: Add/remove target m/z values for EIC extraction
5. **Export**: Download plots as PNG files

## How Deconvolution Works

1. The app sums all MS scans in the selected retention-time window to build one spectrum.
2. The spectrum is smoothed, peaks are detected, and low-intensity peaks below the noise cutoff are ignored.
3. For each anchor peak and charge state, candidate neutral masses are computed with `M = (m/z - H+) * z`.
4. Peaks are grouped into ion sets using MW agreement, abundance cutoff, and charge-ladder contiguity checks.
5. Mass is estimated robustly with ion-level outlier rejection; broad envelopes use per-charge representative ions.
6. Candidate components are selected with overlap and duplicate controls so different species can share part of the same crowded region.
7. A residual second pass runs on unclaimed peaks to recover weaker species.
8. Optional singly charged (`z=1`) ions can be merged into results when enabled.

## Time Progression Colors

- Grey: Initial (t=0)
- Blue: Mid timepoint
- Red: Overnight

## Troubleshooting

### Network drive not accessible

Make sure the network drive is mounted before starting Docker:
```bash
# macOS
mount -t smbfs //server/share /Volumes/chab_loc_lang_s1
```

Then update `docker-compose.yml` with the correct path.

### rainbow-api errors

The app uses [rainbow-api](https://github.com/evanyeyeye/rainbow) for reading `.D` data. If you encounter issues:

```bash
pip install --upgrade rainbow-api
```
