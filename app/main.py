"""LC-MS Analysis Web Application - Streamlit Entry Point."""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="LC-MS Analysis",
    page_icon=":material/science:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Minimal imports needed for loading screen
from pathlib import Path
import os

# ETH Zurich blue color
ETH_BLUE = "#215CAF"

# Loading overlay CSS/HTML
LOADING_OVERLAY = f"""
<style>
    .loading-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: white;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 999999;
    }}
    .spinner {{
        width: 60px;
        height: 60px;
        position: relative;
        animation: spinner-rotate 1.2s linear infinite;
    }}
    .spinner-dot {{
        position: absolute;
        width: 10px;
        height: 10px;
        background: {ETH_BLUE};
        border-radius: 50%;
        animation: spinner-fade 1.2s linear infinite;
    }}
    .spinner-dot:nth-child(1) {{ top: 0; left: 25px; animation-delay: 0s; }}
    .spinner-dot:nth-child(2) {{ top: 7px; left: 43px; animation-delay: -0.15s; }}
    .spinner-dot:nth-child(3) {{ top: 25px; left: 50px; animation-delay: -0.3s; }}
    .spinner-dot:nth-child(4) {{ top: 43px; left: 43px; animation-delay: -0.45s; }}
    .spinner-dot:nth-child(5) {{ top: 50px; left: 25px; animation-delay: -0.6s; }}
    .spinner-dot:nth-child(6) {{ top: 43px; left: 7px; animation-delay: -0.75s; }}
    .spinner-dot:nth-child(7) {{ top: 25px; left: 0; animation-delay: -0.9s; }}
    .spinner-dot:nth-child(8) {{ top: 7px; left: 7px; animation-delay: -1.05s; }}
    @keyframes spinner-rotate {{
        100% {{ transform: rotate(360deg); }}
    }}
    @keyframes spinner-fade {{
        0%, 100% {{ opacity: 0.2; transform: scale(0.8); }}
        50% {{ opacity: 1; transform: scale(1); }}
    }}
    .loading-text {{
        margin-top: 24px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 16px;
        color: #333;
    }}
</style>
<div class="loading-overlay">
    <div class="spinner">
        <div class="spinner-dot"></div>
        <div class="spinner-dot"></div>
        <div class="spinner-dot"></div>
        <div class="spinner-dot"></div>
        <div class="spinner-dot"></div>
        <div class="spinner-dot"></div>
        <div class="spinner-dot"></div>
        <div class="spinner-dot"></div>
    </div>
    <div class="loading-text">Please wait...</div>
</div>
"""

# Show loading overlay immediately
_loading_placeholder = st.empty()
_loading_placeholder.markdown(LOADING_OVERLAY, unsafe_allow_html=True)


# Lazy load heavy modules using cache_resource
@st.cache_resource(show_spinner=False)
def _init_heavy_modules():
    """Initialize heavy modules once per process."""
    import numpy
    # Clean macOS AppleDouble files (._*) from matplotlib's stylelib before
    # pyplot imports style.core — these break Windows builds with UnicodeDecodeError.
    import matplotlib
    _stylelib = Path(matplotlib.get_data_path()) / 'stylelib'
    for _f in _stylelib.glob('._*'):
        try:
            _f.unlink()
        except OSError:
            pass
    import matplotlib.pyplot as plt
    # Pre-configure matplotlib
    matplotlib.use('Agg')
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 10
    })
    return True


# Initialize heavy modules (cached - only slow on first run)
_init_heavy_modules()

# Now import everything (will be fast since modules are already loaded)
import numpy as np
import matplotlib.pyplot as plt
import csv
from io import StringIO
from html import escape
import zipfile
import tempfile
import shutil

import config
from data_reader import list_d_folders_cached, read_sample_cached, check_rainbow_available
from analysis import (
    extract_eic, smooth_data, calculate_peak_area, find_peaks, find_spectrum_peaks,
    sum_spectra_in_range, deconvolute_protein, deconvolute_protein_local_lcms_machine_like,
    detect_singly_charged, get_theoretical_mz
)
from plotting import (
    create_single_sample_figure,
    create_time_progression_figure,
    create_eic_comparison_figure,
    create_deconvolution_figure,
    create_deconvoluted_masses_figure,
    create_ion_selection_figure,
    create_mass_spectrum_figure,
    create_report_info_page,
    export_figure,
    export_figure_svg,
    export_figure_pdf
)

# Clear loading overlay
_loading_placeholder.empty()

# Custom CSS
st.markdown("""
<style>
    :root { --lcms-sidebar-width: 31.5rem; }
    html, body { height: 100%; overflow: auto !important; }
    body { position: relative; }
    div[data-testid="stAppViewContainer"] { height: 100%; overflow: auto !important; }
    div[data-testid="stApp"] { height: 100%; overflow: auto !important; }
    section[data-testid="stSidebar"] { overflow: auto !important; }
    section[data-testid="stSidebar"] { width: var(--lcms-sidebar-width) !important; }
    section[data-testid="stSidebar"] > div { width: var(--lcms-sidebar-width) !important; }
    @media (min-width: 768px) {
        section[data-testid="stSidebar"][aria-expanded="true"] + div {
            margin-left: var(--lcms-sidebar-width) !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] + div {
            margin-left: 0 !important;
        }
    }
    .main .block-container { overflow: visible !important; }
    .stSelectbox label { font-weight: bold; }
    .main .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    /* Reduce spacing in sidebar */
    section[data-testid="stSidebar"] .stCheckbox { margin-bottom: -10px; }
    section[data-testid="stSidebar"] .stButton { margin-bottom: -5px; }
    section[data-testid="stSidebar"] button { padding: 0.25rem 0.5rem; }
</style>
""", unsafe_allow_html=True)


def is_running_locally():
    """Check if running locally (desktop app) vs cloud."""
    # Check for Streamlit Cloud environment variables
    if os.environ.get('STREAMLIT_SHARING_MODE'):
        return False
    if os.environ.get('STREAMLIT_SERVER_HEADLESS') == 'true':
        # Could be cloud or local headless, check for common cloud indicators
        if os.environ.get('HOME', '').startswith('/home/appuser'):
            return False
    # Check if we have access to local file system
    return os.path.exists('/Users') or os.path.exists('C:\\') or os.path.exists('/home')


def _shift_sci_offset_left(ax, x: float = -0.08) -> None:
    """Move scientific-notation offset text slightly left of the y-axis spine."""
    try:
        offset = ax.yaxis.get_offset_text()
        offset.set_x(x)
        offset.set_ha('left')
    except Exception:
        pass


def init_session_state():
    """Initialize session state variables."""
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []
    if 'loaded_samples' not in st.session_state:
        st.session_state.loaded_samples = {}
    if 'mz_targets' not in st.session_state:
        st.session_state.mz_targets = config.DEFAULT_MZ_VALUES.copy()
    if 'current_path' not in st.session_state or not os.path.exists(st.session_state.current_path):
        st.session_state.current_path = config.BASE_PATH
    if 'uploaded_files_dir' not in st.session_state:
        st.session_state.uploaded_files_dir = None
    if 'data_source' not in st.session_state:
        # Default to 'browse' for local/desktop, 'upload' for cloud
        st.session_state.data_source = 'browse' if is_running_locally() else 'upload'


def handle_file_upload(uploaded_files):
    """Handle uploaded ZIP files containing .D folders."""
    if not uploaded_files:
        return []

    extracted_paths = []

    # Create temp directory for this session if not exists
    if st.session_state.uploaded_files_dir is None or not os.path.exists(st.session_state.uploaded_files_dir):
        st.session_state.uploaded_files_dir = tempfile.mkdtemp(prefix="lcms_")

    temp_dir = st.session_state.uploaded_files_dir

    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.zip'):
                # Extract ZIP file
                zip_path = os.path.join(temp_dir, uploaded_file.name)
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Extract contents (skip macOS metadata)
                extract_dir = os.path.join(temp_dir, uploaded_file.name.replace('.zip', ''))
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        # Skip macOS metadata files (anywhere in path)
                        basename = os.path.basename(member.rstrip('/'))
                        if (member.startswith('__MACOSX') or
                            '__MACOSX/' in member or
                            basename.startswith('._') or
                            basename == '.DS_Store'):
                            continue
                        zip_ref.extract(member, extract_dir)

                # Clean up any remaining macOS metadata
                for root, dirs, files in os.walk(extract_dir):
                    # Remove __MACOSX folders
                    if '__MACOSX' in dirs:
                        shutil.rmtree(os.path.join(root, '__MACOSX'), ignore_errors=True)
                        dirs.remove('__MACOSX')
                    # Remove ._ files and .DS_Store
                    for f in files:
                        if f.startswith('._') or f == '.DS_Store':
                            try:
                                os.remove(os.path.join(root, f))
                            except:
                                pass

                # Find .D folders in extracted contents (use set to avoid duplicates)
                found_d_folders = set()

                # Always walk through looking for .D folders inside the extracted content
                for root, dirs, files in os.walk(extract_dir):
                    for d in dirs:
                        if d.endswith('.D') or d.endswith('.d'):
                            d_path = os.path.abspath(os.path.join(root, d))
                            found_d_folders.add(d_path)
                    # Don't recurse into .D folders
                    dirs[:] = [d for d in dirs if not (d.endswith('.D') or d.endswith('.d'))]

                # If no .D folders found inside, check if extract_dir itself contains the data files
                if not found_d_folders:
                    # Check if this folder has .MS or .ch files (it might be the .D folder contents)
                    try:
                        files_in_extract = os.listdir(extract_dir)
                        has_data_files = any(f.endswith('.MS') or f.endswith('.ch') for f in files_in_extract)
                        if has_data_files:
                            found_d_folders.add(os.path.abspath(extract_dir))
                    except:
                        pass

                extracted_paths.extend(found_d_folders)

                # Clean up zip file
                os.remove(zip_path)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    return extracted_paths


def handle_folder_upload(uploaded_files):
    """Handle uploaded files from folder selection (webkitdirectory)."""
    if not uploaded_files:
        return []

    extracted_paths = []

    # Create temp directory for this session if not exists
    if st.session_state.uploaded_files_dir is None or not os.path.exists(st.session_state.uploaded_files_dir):
        st.session_state.uploaded_files_dir = tempfile.mkdtemp(prefix="lcms_")

    temp_dir = st.session_state.uploaded_files_dir

    # Group files by their .D folder
    d_folders = {}
    for uploaded_file in uploaded_files:
        # File path like "FolderName.D/subfolder/file.ext" or "FolderName.D/file.ext"
        file_path = uploaded_file.name

        # Skip macOS metadata
        if '/__MACOSX/' in file_path or file_path.startswith('__MACOSX'):
            continue
        basename = os.path.basename(file_path)
        if basename.startswith('._') or basename == '.DS_Store':
            continue

        # Find the .D folder in the path
        parts = file_path.replace('\\', '/').split('/')
        d_folder_name = None
        d_folder_idx = -1
        for i, part in enumerate(parts):
            if part.endswith('.D') or part.endswith('.d'):
                d_folder_name = part
                d_folder_idx = i
                break

        if d_folder_name:
            if d_folder_name not in d_folders:
                d_folders[d_folder_name] = []
            # Get relative path within .D folder
            rel_path = '/'.join(parts[d_folder_idx:])
            d_folders[d_folder_name].append((rel_path, uploaded_file))

    # Write files to temp directory
    for d_folder_name, files in d_folders.items():
        d_folder_path = os.path.join(temp_dir, d_folder_name)

        for rel_path, uploaded_file in files:
            file_full_path = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(file_full_path), exist_ok=True)

            with open(file_full_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

        if os.path.isdir(d_folder_path):
            extracted_paths.append(os.path.abspath(d_folder_path))

    return extracted_paths


def sidebar_file_upload():
    """Render file upload interface in sidebar."""
    st.sidebar.header("Upload Data")

    # Upload method selector
    upload_method = st.sidebar.radio(
        "Upload method",
        ["ZIP file", "Folder (direct)"],
        horizontal=True,
        help="ZIP works everywhere. Folder upload works in Chrome/Edge/Firefox."
    )

    if upload_method == "ZIP file":
        st.sidebar.markdown("Upload your `.D` folder as a ZIP file")

        uploaded_files = st.sidebar.file_uploader(
            "Upload .D folder (as ZIP)",
            type=['zip'],
            accept_multiple_files=True,
            key="file_uploader_zip"
        )

        if uploaded_files:
            new_paths = handle_file_upload(uploaded_files)
            for path in new_paths:
                if path not in st.session_state.selected_files:
                    st.session_state.selected_files.append(path)

    else:
        st.sidebar.markdown("""
        **Steps:**
        1. Click 'Browse files' below
        2. Navigate INTO your .D folder
        3. Select ALL files (Ctrl+A / Cmd+A)
        4. Click Open
        """)

        uploaded_files = st.sidebar.file_uploader(
            "Select all files from .D folder",
            accept_multiple_files=True,
            key="file_uploader_folder"
        )

        if uploaded_files:
            new_paths = handle_folder_upload(uploaded_files)
            for path in new_paths:
                if path not in st.session_state.selected_files:
                    st.session_state.selected_files.append(path)

            if new_paths:
                st.sidebar.success(f"Found {len(new_paths)} .D folder(s)")

    # Show debug info for selected files
    if st.session_state.selected_files:
        with st.sidebar.expander("Extracted files (debug)"):
            for d_path in st.session_state.selected_files:
                st.caption(f"**{Path(d_path).name}**")
                st.text(f"Path: {d_path}")
                try:
                    files = list(Path(d_path).iterdir())
                    file_names = [f.name for f in files if f.is_file()]
                    st.text(f"Files ({len(file_names)}): {', '.join(file_names[:20])}")
                    has_ms = any(f.endswith('.MS') for f in file_names)
                    has_uv = any(f.endswith('.ch') for f in file_names)
                    st.text(f"MS: {'Yes' if has_ms else 'No'}, UV: {'Yes' if has_uv else 'No'}")
                except Exception as e:
                    st.text(f"Error listing: {e}")

    # Show uploaded/selected files
    if st.session_state.selected_files:
        st.sidebar.subheader(f"Selected Files ({len(st.session_state.selected_files)})")

        for path in st.session_state.selected_files:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.caption(Path(path).name)
            with col2:
                if st.button("X", key=f"remove_upload_{path}"):
                    st.session_state.selected_files.remove(path)
                    if path in st.session_state.loaded_samples:
                        del st.session_state.loaded_samples[path]
                    st.rerun()

        if st.sidebar.button("Clear all uploads"):
            st.session_state.selected_files = []
            st.session_state.loaded_samples = {}
            # Clean up temp directory
            if st.session_state.uploaded_files_dir and os.path.exists(st.session_state.uploaded_files_dir):
                shutil.rmtree(st.session_state.uploaded_files_dir, ignore_errors=True)
                st.session_state.uploaded_files_dir = None
            st.rerun()

    return st.session_state.selected_files


def get_folder_contents(path: str) -> tuple[list[dict], list[dict]]:
    """Get subfolders and .D folders in a directory."""
    subfolders = []
    d_folders = []

    try:
        p = Path(path)
        for item in p.iterdir():
            try:
                name = item.name
                # Skip hidden files/folders
                if name.startswith('.'):
                    continue

                if item.is_dir():
                    if name.endswith('.D') or name.endswith('.d'):
                        # This is a .D data folder
                        try:
                            stat = item.stat()
                            d_folders.append({
                                'path': str(item),
                                'name': name,
                                'date': stat.st_mtime,
                            })
                        except (OSError, PermissionError):
                            pass
                    else:
                        # Regular subfolder
                        try:
                            stat = item.stat()
                            subfolders.append({
                                'path': str(item),
                                'name': name,
                                'date': stat.st_mtime,
                            })
                        except (OSError, PermissionError):
                            subfolders.append({
                                'path': str(item),
                                'name': name,
                                'date': 0.0,
                            })
            except (OSError, PermissionError):
                # Skip files/folders we can't access
                continue
    except (OSError, PermissionError) as e:
        st.sidebar.error(f"Cannot access folder: {e}")

    # Sort results
    subfolders.sort(key=lambda x: x['name'].lower())
    d_folders.sort(key=lambda x: x['name'].lower())

    return subfolders, d_folders


def extract_apex_spectrum(sample, start_time: float, end_time: float, n_scans: int = 1):
    """
    Extract spectrum around TIC apex within the time window.

    Args:
        sample: SampleData object
        start_time: Start of time window (minutes)
        end_time: End of time window (minutes)
        n_scans: Number of scans to average around apex (1 = single scan, 3-5 recommended)

    Returns:
        Tuple of (mz_array, intensity_array)
    """
    if sample.ms_scans is None or sample.ms_times is None:
        return np.array([]), np.array([])

    time_mask = (sample.ms_times >= start_time) & (sample.ms_times <= end_time)
    scan_indices = np.where(time_mask)[0]
    if len(scan_indices) == 0:
        return np.array([]), np.array([])

    # Find apex scan (highest TIC)
    apex_idx = None
    max_sum = -1.0
    for idx in scan_indices:
        scan = sample.ms_scans[idx]
        if scan is None:
            continue
        total = float(np.sum(scan)) if isinstance(scan, np.ndarray) else 0.0
        if total > max_sum:
            max_sum = total
            apex_idx = idx

    if apex_idx is None:
        return np.array([]), np.array([])

    # If n_scans == 1, return single apex scan (original behavior)
    if n_scans <= 1:
        if sample.ms_mz_axis is not None:
            return sample.ms_mz_axis, sample.ms_scans[apex_idx]
        return sum_spectra_in_range(sample, start_time, end_time)

    # Otherwise, average n_scans centered on apex
    half = n_scans // 2

    # Find valid range within time window
    apex_pos = np.where(scan_indices == apex_idx)[0][0]
    start_pos = max(0, apex_pos - half)
    end_pos = min(len(scan_indices) - 1, apex_pos + half)

    # Get the actual scan indices to average
    scans_to_avg = scan_indices[start_pos:end_pos + 1]

    if len(scans_to_avg) == 0:
        return np.array([]), np.array([])

    # Average the scans
    if sample.ms_mz_axis is not None:
        mz_axis = sample.ms_mz_axis
        summed_intensities = np.zeros(len(mz_axis))
        count = 0
        for idx in scans_to_avg:
            scan = sample.ms_scans[idx]
            if scan is not None and isinstance(scan, np.ndarray):
                summed_intensities += scan
                count += 1
        if count > 0:
            summed_intensities /= count  # Average, not sum
        return mz_axis, summed_intensities

    # Fallback: use sum_spectra_in_range for the apex window
    apex_start_time = float(sample.ms_times[scans_to_avg[0]])
    apex_end_time = float(sample.ms_times[scans_to_avg[-1]])
    mz, intensity = sum_spectra_in_range(sample, apex_start_time, apex_end_time)
    # Normalize to average
    if len(scans_to_avg) > 1:
        intensity = intensity / len(scans_to_avg)
    return mz, intensity


def _detect_deconvolution_window(sample) -> tuple[float, float]:
    """Auto-detect the default deconvolution time window for a sample."""
    if sample.ms_times is None or len(sample.ms_times) == 0:
        return 0.0, 0.0

    min_time = float(sample.ms_times[0])
    max_time = float(sample.ms_times[-1])

    if sample.tic is None:
        return min_time, min(min_time + 1.0, max_time)

    tic_smoothed = smooth_data(sample.tic, 5)
    peaks = find_peaks(sample.ms_times, tic_smoothed, height_threshold=0.3, prominence=0.1)
    if not peaks:
        return min_time, min(min_time + 1.0, max_time)

    # For C4 methods, ignore early void/injection peaks when possible.
    is_c4 = getattr(sample, "is_c4_method", False)
    if is_c4:
        protein_peaks = [p for p in peaks if p["time"] >= 1.8]
        if protein_peaks:
            peaks = protein_peaks

    # Use the dominant peak only, then tighten boundaries to 48% of its height.
    # This avoids broad auto-windows from long low-slope tails.
    dominant_peak = max(peaks, key=lambda p: p["intensity"])

    threshold_left = dominant_peak["intensity"] * 0.48
    left_idx = dominant_peak["index"]
    while left_idx > 0 and tic_smoothed[left_idx] > threshold_left:
        left_idx -= 1

    threshold_right = dominant_peak["intensity"] * 0.48
    right_idx = dominant_peak["index"]
    while right_idx < len(tic_smoothed) - 1 and tic_smoothed[right_idx] > threshold_right:
        right_idx += 1

    auto_start = float(sample.ms_times[left_idx])
    auto_end = float(sample.ms_times[right_idx])

    # Keep default auto window narrow and centered for deconvolution quality.
    max_window_width = 0.30
    if auto_end > auto_start and (auto_end - auto_start) > max_window_width:
        peak_time = float(sample.ms_times[dominant_peak["index"]])
        half_width = max_window_width / 2.0
        auto_start = max(min_time, peak_time - half_width)
        auto_end = min(max_time, peak_time + half_width)

    return auto_start, auto_end


def _run_local_machine_protein_profile(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    min_charge: int,
    max_charge: int,
    min_peaks: int,
    noise_cutoff: float,
    abundance_cutoff_pct: float,
    mw_agreement_pct: float,
    mw_assign_cutoff_pct: float,
    envelope_cutoff_pct: float,
    pwhh: float,
    low_mw: float,
    high_mw: float,
    contig_min: int,
    use_mz_agreement: bool,
    use_monoisotopic: bool,
) -> list[dict]:
    """Run one protein-profile deconvolution pass with the provided charge bounds."""
    return deconvolute_protein_local_lcms_machine_like(
        mz,
        intensity,
        min_charge=min_charge,
        max_charge=max_charge,
        min_peaks=min_peaks,
        noise_cutoff=noise_cutoff,
        abundance_cutoff=abundance_cutoff_pct / 100.0,
        mw_agreement=mw_agreement_pct / 100.0,
        mw_assign_cutoff=mw_assign_cutoff_pct / 100.0,
        envelope_cutoff=envelope_cutoff_pct / 100.0,
        pwhh=pwhh,
        low_mw=low_mw,
        high_mw=high_mw,
        contig_min=contig_min,
        use_mz_agreement=use_mz_agreement,
        use_monoisotopic_proton=use_monoisotopic,
    )


def _run_local_machine_auto_profile(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    min_charge: int,
    max_charge: int,
    min_peaks: int,
    noise_cutoff: float,
    abundance_cutoff_pct: float,
    mw_agreement_pct: float,
    mw_assign_cutoff_pct: float,
    envelope_cutoff_pct: float,
    pwhh: float,
    low_mw: float,
    high_mw: float,
    contig_min: int,
    use_mz_agreement: bool,
    use_monoisotopic: bool,
    allow_auto_fallback: bool = False,
) -> tuple[list[dict], str]:
    """Run local-machine deconvolution (single pass)."""
    results = _run_local_machine_protein_profile(
        mz,
        intensity,
        min_charge=min_charge,
        max_charge=max_charge,
        min_peaks=min_peaks,
        noise_cutoff=noise_cutoff,
        abundance_cutoff_pct=abundance_cutoff_pct,
        mw_agreement_pct=mw_agreement_pct,
        mw_assign_cutoff_pct=mw_assign_cutoff_pct,
        envelope_cutoff_pct=envelope_cutoff_pct,
        pwhh=pwhh,
        low_mw=low_mw,
        high_mw=high_mw,
        contig_min=contig_min,
        use_mz_agreement=use_mz_agreement,
        use_monoisotopic=use_monoisotopic,
    )
    return results, ""


def _run_default_deconvolution(mz: np.ndarray, intensity: np.ndarray) -> list[dict]:
    """Run deconvolution with the default non-expert settings."""
    # Defaults aligned with the main deconvolution view.
    pwhh = 0.6
    mw_agreement_pct = 0.02
    noise_cutoff = 1000.0
    abundance_cutoff_pct = 5.0
    mw_assign_cutoff_pct = 40.0
    envelope_cutoff_pct = 50.0
    contig_min = 3
    low_mw = 500.0
    high_mw = 50000.0
    min_charge = 1
    max_charge = 50

    results, _ = _run_local_machine_auto_profile(
        mz,
        intensity,
        min_charge=min_charge,
        max_charge=max_charge,
        min_peaks=3,
        noise_cutoff=noise_cutoff,
        abundance_cutoff_pct=abundance_cutoff_pct,
        mw_agreement_pct=mw_agreement_pct,
        mw_assign_cutoff_pct=mw_assign_cutoff_pct,
        envelope_cutoff_pct=envelope_cutoff_pct,
        pwhh=pwhh,
        low_mw=low_mw,
        high_mw=high_mw,
        contig_min=contig_min,
        use_mz_agreement=False,
        use_monoisotopic=False,
        allow_auto_fallback=True,
    )

    results = [r for r in results if low_mw <= r["mass"] <= high_mw]
    results.sort(key=lambda x: x["intensity"], reverse=True)
    return results


def _is_likely_half_mass_alias(component: dict, all_results: list[dict], ratio_tol: float = 0.01) -> bool:
    """Detect likely M/2 harmonic aliases for display filtering."""
    charge_states = component.get("charge_states", [])
    if not charge_states:
        return False

    if int(component.get("num_charges", 0)) > 3:
        return False
    if max(charge_states) > 6:
        return False

    mass = float(component.get("mass", 0.0))
    if mass <= 0:
        return False

    for other in all_results:
        if other is component:
            continue
        other_mass = float(other.get("mass", 0.0))
        if other_mass <= mass:
            continue
        if abs((other_mass / mass) - 2.0) <= ratio_tol:
            return True
    return False


def _filter_display_deconvolution_results(
    results: list[dict],
    *,
    expert_mode: bool,
    min_rel_intensity: float = 0.05,
) -> list[dict]:
    """Apply display-only filtering for non-expert deconvolution views."""
    if not results:
        return []

    ordered = sorted(results, key=lambda r: float(r.get("intensity", 0.0)), reverse=True)
    top_intensity = float(ordered[0].get("intensity", 0.0))
    if top_intensity <= 0:
        return ordered

    filtered = [r for r in ordered if float(r.get("intensity", 0.0)) >= min_rel_intensity * top_intensity]
    if expert_mode:
        return filtered

    # Non-expert mode: hide likely harmonic aliases that can outrank true species.
    return [r for r in filtered if not _is_likely_half_mass_alias(r, ordered)]


def render_text_table(rows: list[dict], columns: list[str], max_lines: int = 0) -> None:
    """Render a simple ASCII table to avoid pyarrow dependency.

    Args:
        rows: List of row dictionaries
        columns: Column names
        max_lines: If > 0, show scrollable container with this many visible lines
    """
    if not rows:
        st.info("No results to display.")
        return
    col_widths = []
    for col in columns:
        max_len = len(col)
        for row in rows:
            max_len = max(max_len, len(str(row.get(col, ""))))
        col_widths.append(max_len)

    def fmt_row(vals):
        return " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    header = fmt_row(columns)
    sep = "-+-".join("-" * w for w in col_widths)
    lines = [header, sep]
    for row in rows:
        lines.append(fmt_row([row.get(col, "") for col in columns]))

    if max_lines > 0 and len(rows) > max_lines:
        # Calculate height: ~1.4em per line, plus header and separator
        height_px = int((max_lines + 2) * 22)  # ~22px per line in code block
        st.markdown(
            f'<div style="max-height: {height_px}px; overflow-y: auto;">'
            f'<pre style="margin: 0; padding: 0.5em; background: #262730; color: #fafafa; '
            f'border-radius: 0.25rem; font-size: 14px;">{chr(10).join(lines)}</pre></div>',
            unsafe_allow_html=True
        )
    else:
        st.code("\n".join(lines), language="text")


def render_html_table(rows: list[dict], columns: list[str] | None = None, max_lines: int = 0) -> None:
    """Render a clean HTML table without pyarrow/pandas dependency."""
    if not rows:
        st.info("No results to display.")
        return

    if columns is None:
        columns = list(rows[0].keys())

    header_html = "".join(
        f'<th style="text-align:left; padding:0.45rem 0.6rem; border-bottom:1px solid #666;">{escape(str(col))}</th>'
        for col in columns
    )

    body_rows = []
    for row in rows:
        cells = "".join(
            f'<td style="padding:0.40rem 0.6rem; border-bottom:1px solid #3f3f3f;">{escape(str(row.get(col, "")))}</td>'
            for col in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")

    table_html = (
        '<table style="width:100%; border-collapse:collapse; font-size:0.92rem;">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )

    if max_lines > 0 and len(rows) > max_lines:
        row_height_px = 34
        header_height_px = 42
        max_height = header_height_px + max_lines * row_height_px
        st.markdown(
            f'<div style="max-height:{max_height}px; overflow:auto; padding:0;">'
            f"{table_html}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="overflow-x:auto; padding:0;">'
            f"{table_html}</div>",
            unsafe_allow_html=True
        )


def get_windows_drives() -> list[str]:
    """Return a list of available Windows drive roots (e.g., C:\\, D:\\)."""
    if os.name != "nt":
        return []
    drives: list[str] = []
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            drives.append(drive)
    return drives


def sidebar_file_browser():
    """Render the interactive file browser in the sidebar."""

    current_path = st.session_state.current_path
    pending_uncheck = st.session_state.pop("_pending_uncheck_paths", [])
    if isinstance(pending_uncheck, (list, tuple)):
        for path in pending_uncheck:
            checkbox_key = f"select_{path}"
            # Apply queued checkbox resets before widgets are instantiated.
            if checkbox_key in st.session_state:
                st.session_state[checkbox_key] = False

    def _remove_selected_file(path: str) -> None:
        """Remove a selected file and queue its checkbox for uncheck on rerun."""
        if path in st.session_state.selected_files:
            st.session_state.selected_files.remove(path)
        queued = st.session_state.get("_pending_uncheck_paths", [])
        if path not in queued:
            queued.append(path)
        st.session_state["_pending_uncheck_paths"] = queued

    # Keep browser open while selecting multiple files to avoid collapsing
    # after the first checkbox interaction.
    with st.sidebar.expander("File Browser", expanded=True):
        # Path input + Go button
        path_col, go_col = st.columns([4, 1])
        with path_col:
            new_path = st.text_input(
                "Current Path",
                value=current_path,
                key="path_display",
            )
        with go_col:
            if st.button("Go", use_container_width=True):
                if new_path and Path(new_path).exists():
                    st.session_state.current_path = new_path
                    st.rerun()
                else:
                    st.warning(f"Path not found: {new_path}")

        # Quick drive buttons on Windows
        drives = get_windows_drives()
        if drives:
            st.caption("Drives")
            drive_cols = st.columns(min(len(drives), 4))
            for i, drive in enumerate(drives):
                with drive_cols[i % len(drive_cols)]:
                    if st.button(drive, key=f"drive_{drive}", use_container_width=True):
                        st.session_state.current_path = drive
                        st.rerun()

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↑ Up", use_container_width=True):
                parent = str(Path(current_path).parent)
                st.session_state.current_path = parent
                st.rerun()
        with col2:
            if st.button("⌂ Home", use_container_width=True):
                st.session_state.current_path = config.BASE_PATH
                st.rerun()

        # Check if path exists
        if not Path(current_path).exists():
            st.warning(f"Path not found: {current_path}")
            st.info("Make sure the network drive is mounted.")
            return []

        # Get contents
        subfolders, d_folders = get_folder_contents(current_path)

        # Show subfolders as clickable buttons
        if subfolders:
            st.caption("Folders")
            folder_sort_by = st.selectbox(
                "Sort by",
                ["Date (Newest)", "Date (Oldest)", "Name (A-Z)", "Name (Z-A)"],
                key="sort_folders_option"
            )
            if folder_sort_by == "Name (A-Z)":
                subfolders.sort(key=lambda x: x['name'].lower())
            elif folder_sort_by == "Name (Z-A)":
                subfolders.sort(key=lambda x: x['name'].lower(), reverse=True)
            elif folder_sort_by == "Date (Newest)":
                subfolders.sort(key=lambda x: x.get('date', 0), reverse=True)
            elif folder_sort_by == "Date (Oldest)":
                subfolders.sort(key=lambda x: x.get('date', 0))
            for folder in subfolders:
                if st.button(f"{folder['name']}", key=f"folder_{folder['path']}", use_container_width=True):
                    st.session_state.current_path = folder['path']
                    st.rerun()

        # Show .D folders for selection
        if d_folders:
            st.caption(f"Data Files ({len(d_folders)})")

            # Sort options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Name (A-Z)", "Name (Z-A)"], key="sort_option")

            if sort_by == "Name (A-Z)":
                d_folders.sort(key=lambda x: x['name'].lower())
            elif sort_by == "Name (Z-A)":
                d_folders.sort(key=lambda x: x['name'].lower(), reverse=True)
            elif sort_by == "Date (Newest)":
                d_folders.sort(key=lambda x: x['date'], reverse=True)
            elif sort_by == "Date (Oldest)":
                d_folders.sort(key=lambda x: x['date'])

            # Keep the selectable file list scrollable (max ~20 visible rows).
            max_visible_files = 20
            row_height_px = 34
            list_height = max(120, min(len(d_folders), max_visible_files) * row_height_px)
            with st.container(height=list_height):
                # Checkbox for each .D folder
                for d_folder in d_folders:
                    is_selected = d_folder['path'] in st.session_state.selected_files
                    if st.checkbox(
                        f"{d_folder['name']}",
                        value=is_selected,
                        key=f"select_{d_folder['path']}"
                    ):
                        if d_folder['path'] not in st.session_state.selected_files:
                            st.session_state.selected_files.append(d_folder['path'])
                    else:
                        if d_folder['path'] in st.session_state.selected_files:
                            st.session_state.selected_files.remove(d_folder['path'])

        if not subfolders and not d_folders:
            st.info("No folders or .D files found here")

    # Show selected files summary in expander
    if st.session_state.selected_files:
        with st.sidebar.expander(f"Selected Files ({len(st.session_state.selected_files)})", expanded=True):
            for path in list(st.session_state.selected_files):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(Path(path).name)
                with col2:
                    if st.button("X", key=f"remove_{path}"):
                        _remove_selected_file(path)
                        st.rerun()

            if st.button("Clear all", use_container_width=True):
                for path in list(st.session_state.selected_files):
                    _remove_selected_file(path)
                st.session_state.selected_files = []
                st.rerun()

    return st.session_state.selected_files


def sidebar_settings():
    """Render settings in sidebar."""
    st.sidebar.header("Settings")

    # Graph/Export settings (collapsible, at top)
    with st.sidebar.expander("Graph & Export Settings"):
        # DPI
        export_dpi = st.slider(
            "Export DPI",
            min_value=72,
            max_value=600,
            value=config.EXPORT_DPI,
            step=50,
            help="Resolution for PNG export"
        )

        # Figure dimensions
        fig_width = st.slider(
            "Figure width (inches)",
            min_value=4,
            max_value=16,
            value=6,
            step=1
        )
        fig_height_per_panel = st.slider(
            "Height per panel (inches)",
            min_value=1,
            max_value=5,
            value=2,
            step=1
        )

        # Line settings
        line_width = st.slider(
            "Line width",
            min_value=0.5,
            max_value=3.0,
            value=0.8,
            step=0.1
        )

        # Grid
        show_grid = st.checkbox("Show grid", value=False)

        st.caption("Deconvoluted Mass Axis (Da)")
        col1, col2 = st.columns(2)
        with col1:
            deconv_x_min_da = st.number_input(
                "Min",
                min_value=100.0,
                max_value=1_000_000.0,
                value=1000.0,
                step=100.0
            )
        with col2:
            deconv_x_max_da = st.number_input(
                "Max",
                min_value=100.0,
                max_value=1_000_000.0,
                value=50000.0,
                step=100.0
            )

        if deconv_x_max_da <= deconv_x_min_da:
            st.warning("Deconvoluted mass axis max must be greater than min. Using min + 1000 Da.")
            deconv_x_max_da = deconv_x_min_da + 1000.0

        st.caption("Time progression colors:")
        col1, col2, col3 = st.columns(3)
        with col1:
            color_initial = st.color_picker("Initial", config.TIME_COLORS["initial"])
        with col2:
            color_mid = st.color_picker("Mid", config.TIME_COLORS["mid"])
        with col3:
            color_final = st.color_picker("Final", config.TIME_COLORS["final"])

    # Labels & Titles settings
    with st.sidebar.expander("Labels & Titles"):
        # Main titles
        title_single = st.text_input("Single sample title", value="Sample: {name}", help="Use {name} for sample name")
        title_progression = st.text_input("Time progression title", value="Time Progression Analysis")

        # Axis labels
        x_label = st.text_input("X-axis label", value="Time (min)")

        st.caption("Y-axis labels:")
        y_label_uv = st.text_input("UV Y-axis", value="UV {wavelength}nm (mAU)", help="Use {wavelength} for wavelength value")
        y_label_tic = st.text_input("TIC Y-axis", value="TIC Intensity")
        y_label_eic = st.text_input("EIC Y-axis", value="EIC Intensity")

        st.caption("Panel titles:")
        panel_title_uv = st.text_input("UV panel", value="UV Chromatogram ({wavelength} nm)")
        panel_title_tic = st.text_input("TIC panel", value="Total Ion Chromatogram (TIC)")
        panel_title_eic = st.text_input("EIC panel", value="EIC m/z {mz} (±{window})", help="Use {mz} and {window}")

        st.caption("Deconvoluted masses figure:")
        deconv_show_title = st.checkbox("Show title", value=True, key="deconv_show_title")
        deconv_show_subtitle = st.checkbox("Show sample name", value=True, key="deconv_show_subtitle")

    # Ionization mode
    ion_mode = st.sidebar.radio(
        "Ionization Mode",
        ["Positive (+)", "Negative (-)"],
        horizontal=True,
        help="Select MS ionization mode"
    )

    # UV wavelength - show checkboxes based on available wavelengths
    st.sidebar.markdown("**UV Wavelengths**")

    # Get available wavelengths from loaded samples
    available_wavelengths = set()
    for sample in st.session_state.loaded_samples.values():
        if hasattr(sample, 'uv_wavelengths') and sample.uv_wavelengths is not None:
            for wl in sample.uv_wavelengths:
                try:
                    available_wavelengths.add(float(wl))
                except (ValueError, TypeError):
                    pass

    # Initialize selected wavelengths in session state
    if 'selected_wavelengths' not in st.session_state:
        st.session_state.selected_wavelengths = []

    selected_wavelengths = []
    if available_wavelengths:
        sorted_wls = sorted(available_wavelengths)
        for wl in sorted_wls:
            # Default: select 194nm if available, otherwise first wavelength
            default_selected = (wl == 194.0) or (wl == sorted_wls[0] and 194.0 not in available_wavelengths)
            if st.sidebar.checkbox(f"{wl:.0f} nm", value=default_selected, key=f"uv_wl_{wl}"):
                selected_wavelengths.append(wl)
    else:
        st.sidebar.caption("Load a sample to see available wavelengths")
        selected_wavelengths = [config.UV_WAVELENGTH]  # fallback

    # Use first selected or default
    uv_wl = selected_wavelengths[0] if selected_wavelengths else config.UV_WAVELENGTH

    # Smoothing settings
    with st.sidebar.expander("Smoothing"):
        uv_smooth = st.slider(
            "UV smoothing",
            min_value=1,
            max_value=99,
            value=config.UV_SMOOTHING_WINDOW,
            step=2
        )
        eic_smooth = st.slider(
            "EIC smoothing",
            min_value=1,
            max_value=49,
            value=config.EIC_SMOOTHING_WINDOW,
            step=2
        )

    # m/z targets in sidebar
    with st.sidebar.expander("Target m/z Values", expanded=True):
        mz_window = st.slider(
            "m/z window (±)",
            min_value=0.1,
            max_value=2.0,
            value=config.DEFAULT_MZ_WINDOW,
            step=0.1
        )

        # Quick add
        new_mz = st.number_input(
            "Add m/z",
            min_value=0.0,
            max_value=2000.0,
            value=100.0,
            step=0.01,
            key="new_mz_input"
        )
        if st.button("+ Add", key="add_mz_btn"):
            if new_mz not in st.session_state.mz_targets:
                st.session_state.mz_targets.append(new_mz)
                st.rerun()

        # Show current targets
        st.caption("Current targets:")
        for i, mz in enumerate(st.session_state.mz_targets):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"m/z {mz:.2f}")
            with col2:
                if st.button("X", key=f"rm_mz_{i}"):
                    st.session_state.mz_targets.pop(i)
                    st.rerun()

        if st.button("Reset to defaults"):
            st.session_state.mz_targets = config.DEFAULT_MZ_VALUES.copy()
            st.rerun()

    return {
        'uv_wavelength': uv_wl,
        'uv_wavelengths': selected_wavelengths,  # all selected wavelengths
        'uv_smoothing': uv_smooth,
        'eic_smoothing': eic_smooth,
        'mz_window': mz_window,
        'ion_mode': 'positive' if 'Positive' in ion_mode else 'negative',
        'export_dpi': export_dpi,
        'fig_width': fig_width,
        'fig_height_per_panel': fig_height_per_panel,
        'line_width': line_width,
        'show_grid': show_grid,
        'deconv_x_min_da': deconv_x_min_da,
        'deconv_x_max_da': deconv_x_max_da,
        'deconv_show_title': deconv_show_title,
        'deconv_show_subtitle': deconv_show_subtitle,
        'y_scale': 'linear',
        'colors': {
            'initial': color_initial,
            'mid': color_mid,
            'final': color_final
        },
        'labels': {
            'title_single': title_single,
            'title_progression': title_progression,
            'x_label': x_label,
            'y_label_uv': y_label_uv,
            'y_label_tic': y_label_tic,
            'y_label_eic': y_label_eic,
            'panel_title_uv': panel_title_uv,
            'panel_title_tic': panel_title_tic,
            'panel_title_eic': panel_title_eic
        }
    }


def load_samples(file_paths: list[str]) -> dict:
    """Load selected samples."""
    samples = {}
    for path in file_paths:
        if path not in st.session_state.loaded_samples:
            with st.spinner(f"Loading {Path(path).name}..."):
                sample = read_sample_cached(path)
                st.session_state.loaded_samples[path] = sample
        samples[path] = st.session_state.loaded_samples[path]
    return samples


def single_sample_analysis(sample, settings):
    """Display single sample analysis view."""
    st.header(f"Single Sample: {sample.name}")

    if sample.error:
        st.error(f"Error loading sample: {sample.error}")
        # Show debug info even on error
        with st.expander("Debug Info"):
            st.write(f"Path: {sample.folder_path}")
            if hasattr(sample, '_debug_info'):
                st.json(sample._debug_info)
        return

    # Sample info
    col1, col2, col3 = st.columns(3)
    with col1:
        has_uv = sample.uv_data is not None
        st.metric("UV Data", "Available" if has_uv else "Not found")
    with col2:
        has_ms = sample.ms_scans is not None
        st.metric("MS Data", "Available" if has_ms else "Not found")
    with col3:
        if sample.ms_times is not None:
            st.metric("MS Scans", len(sample.ms_times))

    # Generate figures in two columns
    st.divider()
    mz_targets = st.session_state.mz_targets
    style = {
        'fig_width': settings['fig_width'],
        'fig_height_per_panel': settings['fig_height_per_panel'],
        'line_width': settings['line_width'],
        'show_grid': settings['show_grid'],
        'y_scale': settings['y_scale'],
        'colors': settings['colors'],
        'labels': settings['labels']
    }

    # Two column layout for plots
    left_col, right_col = st.columns(2)

    with left_col:
        # UV Chromatograms - one panel per selected wavelength
        if sample.uv_data is not None:
            selected_wavelengths = settings.get('uv_wavelengths') or [settings['uv_wavelength']]
            for wl in selected_wavelengths:
                fig_uv, ax_uv = plt.subplots(figsize=(5, 2.5))
                uv_data = sample.get_uv_at_wavelength(wl)
                if uv_data is not None:
                    plot_data = smooth_data(uv_data, settings['uv_smoothing'])
                    ax_uv.plot(sample.uv_times, plot_data, linewidth=settings['line_width'])
                    ax_uv.set_xlabel("Time (min)", fontsize=8)
                    ax_uv.set_ylabel(f"UV {wl:.0f}nm (mAU)", fontsize=8)
                    ax_uv.set_title(f"UV Chromatogram ({wl:.0f} nm)", fontsize=9)
                    ax_uv.tick_params(labelsize=7)
                else:
                    ax_uv.text(0.5, 0.5, f"No UV data at {wl:.0f} nm", ha='center', va='center', transform=ax_uv.transAxes)
                fig_uv.tight_layout()
                st.pyplot(fig_uv, use_container_width=True)
                plt.close(fig_uv)

        # TIC
        if sample.tic is not None and sample.ms_times is not None:
            fig_tic, ax_tic = plt.subplots(figsize=(5, 2.5))
            ax_tic.plot(sample.ms_times, sample.tic, linewidth=settings['line_width'])
            ax_tic.set_xlabel("Time (min)", fontsize=8)
            ax_tic.set_ylabel("TIC Intensity", fontsize=8)
            ax_tic.set_title("Total Ion Chromatogram (TIC)", fontsize=9)
            ax_tic.tick_params(labelsize=7)
            fig_tic.tight_layout()
            st.pyplot(fig_tic, use_container_width=True)
            plt.close(fig_tic)

    with right_col:
        # EICs
        if mz_targets and sample.ms_scans is not None:
            for mz in mz_targets:
                fig_eic, ax_eic = plt.subplots(figsize=(5, 2.5))
                eic = extract_eic(sample, mz, settings['mz_window'])
                if eic is not None:
                    plot_data = smooth_data(eic, settings['eic_smoothing'])
                    ax_eic.plot(sample.ms_times, plot_data, linewidth=settings['line_width'])
                    ax_eic.set_xlabel("Time (min)", fontsize=8)
                    ax_eic.set_ylabel("EIC Intensity", fontsize=8)
                    ax_eic.set_title(f"EIC m/z {mz:.2f} (±{settings['mz_window']})", fontsize=9)
                    ax_eic.tick_params(labelsize=7)
                    ax_eic.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0), useMathText=True)
                    _shift_sci_offset_left(ax_eic)
                else:
                    ax_eic.text(0.5, 0.5, f"No data for m/z {mz}", ha='center', va='center', transform=ax_eic.transAxes)
                fig_eic.tight_layout()
                st.pyplot(fig_eic, use_container_width=True)
                plt.close(fig_eic)

    # Combined figure for export
    selected_wavelengths = settings.get('uv_wavelengths') or [settings['uv_wavelength']]
    fig = create_single_sample_figure(
        sample,
        uv_wavelengths=selected_wavelengths,
        eic_targets=mz_targets,
        style=style,
        mz_window=settings['mz_window'],
        uv_smoothing=settings['uv_smoothing'],
        eic_smoothing=settings['eic_smoothing']
    )

    # Export buttons
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        png_data = export_figure(fig, dpi=settings['export_dpi'])
        st.download_button(
            label="Download PNG (combined)",
            data=png_data,
            file_name=f"{sample.name}_analysis.png",
            mime="image/png"
        )
    with col2:
        svg_data = export_figure_svg(fig)
        st.download_button(
            label="Download SVG",
            data=svg_data,
            file_name=f"{sample.name}_analysis.svg",
            mime="image/svg+xml"
        )
    with col3:
        pdf_data = export_figure_pdf(fig, dpi=settings['export_dpi'])
        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name=f"{sample.name}_analysis.pdf",
            mime="application/pdf"
        )
    plt.close(fig)

    # Show debug info at bottom
    with st.expander("Data Debug Info"):
        if hasattr(sample, '_debug_info') and sample._debug_info:
            st.json(sample._debug_info)

        # Show available wavelengths
        if sample.uv_wavelengths is not None:
            try:
                wl_list = [float(w) for w in sample.uv_wavelengths]
                st.write(f"UV wavelengths: {', '.join([f'{w:.0f}' for w in wl_list[:20]])}")
            except:
                st.write(f"UV wavelengths (raw): {list(sample.uv_wavelengths[:10])}")

        # Show MS scan info
        if sample.ms_scans is not None and len(sample.ms_scans) > 0:
            scan = sample.ms_scans[0]
            if scan is not None:
                st.write(f"First scan type: {type(scan).__name__}")
                if hasattr(scan, 'mz'):
                    st.write(f"  Has 'mz' attr: {len(scan.mz)} values, range: {min(scan.mz):.1f} - {max(scan.mz):.1f}")
                if hasattr(scan, 'masses'):
                    st.write(f"  Has 'masses' attr: {len(scan.masses)} values")
                if hasattr(scan, 'intensity'):
                    st.write(f"  Has 'intensity' attr: {len(scan.intensity)} values")
                if hasattr(scan, 'intensities'):
                    st.write(f"  Has 'intensities' attr: {len(scan.intensities)} values")
                if isinstance(scan, np.ndarray):
                    st.write(f"  Array shape: {scan.shape}")


def time_progression_analysis(samples: list, settings):
    """Display time progression comparison view."""
    st.header("Time Progression Analysis")

    if len(samples) < 2:
        st.warning("Select at least 2 samples for time progression analysis")
        return

    # Check for errors
    for sample in samples:
        if sample.error:
            st.error(f"Error loading {sample.name}: {sample.error}")
            return

    # Allow explicit mapping of samples to progression roles.
    if len(samples) == 2:
        roles = ["Initial (t=0)", "Overnight"]
    else:
        roles = ["Initial (t=0)", "Mid timepoint", "Overnight"]
        for i in range(3, len(samples)):
            roles.append(f"Sample {i + 1}")
    sample_names = [s.name for s in samples]

    st.subheader("Sample Assignment")
    selected_indices: list[int] = []
    assign_cols = st.columns(len(roles))
    for i, (col, role) in enumerate(zip(assign_cols, roles)):
        with col:
            default_idx = min(i, len(sample_names) - 1)
            selected_name = st.selectbox(
                f"{role} sample",
                sample_names,
                index=default_idx,
                key=f"time_role_{i}"
            )
            selected_indices.append(sample_names.index(selected_name))

    if len(set(selected_indices)) != len(selected_indices):
        st.warning("Choose a different sample for each role.")
        return

    ordered_samples = [samples[i] for i in selected_indices]

    # Label inputs (for legend text)
    st.subheader("Sample Labels")
    labels = []
    label_cols = st.columns(len(ordered_samples))
    for i, (col, sample, role) in enumerate(zip(label_cols, ordered_samples, roles)):
        with col:
            label = st.text_input(
                f"Label for {sample.name}",
                value=role,
                key=f"label_tp_{i}"
            )
            labels.append(label)

    # Generate figure
    st.divider()
    mz_targets = st.session_state.mz_targets
    style = {
        'fig_width': settings['fig_width'],
        'fig_height_per_panel': settings['fig_height_per_panel'],
        'line_width': settings['line_width'],
        'show_grid': settings['show_grid'],
        'y_scale': settings['y_scale'],
        'colors': settings['colors'],
        'labels': settings['labels']
    }

    methods = [getattr(s, "acq_method", None) for s in ordered_samples]
    methods_present = [m for m in methods if m]
    if methods_present:
        unique_methods = sorted(set(methods_present))
        method_text = unique_methods[0] if len(unique_methods) == 1 else ", ".join(unique_methods)
    else:
        method_text = "n/a"

    file_names = [s.name[:-2] if s.name.lower().endswith(".d") else s.name for s in ordered_samples]
    files_text = ", ".join(file_names)

    fig = create_time_progression_figure(
        ordered_samples,
        labels,
        uv_wavelength=settings['uv_wavelength'],
        eic_targets=mz_targets,
        style=style,
        mz_window=settings['mz_window'],
        uv_smoothing=settings['uv_smoothing'],
        eic_smoothing=settings['eic_smoothing'],
        metadata_method=method_text,
        metadata_files=files_text
    )

    # Display
    st.pyplot(fig, use_container_width=True)

    # Export buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        png_data = export_figure(fig, dpi=settings['export_dpi'])
        st.download_button(
            label="Download PNG",
            data=png_data,
            file_name="time_progression_analysis.png",
            mime="image/png"
        )
    with col2:
        svg_data = export_figure_svg(fig)
        st.download_button(
            label="Download SVG",
            data=svg_data,
            file_name="time_progression_analysis.svg",
            mime="image/svg+xml"
        )
    with col3:
        pdf_data = export_figure_pdf(fig, dpi=settings['export_dpi'])
        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name="time_progression_analysis.pdf",
            mime="application/pdf"
        )

    # Metadata is embedded in the exported figure; no duplicate footer needed.


def eic_batch_analysis(sample, settings):
    """Display EIC batch extraction view."""
    st.header(f"EIC Batch Analysis: {sample.name}")

    if sample.error:
        st.error(f"Error loading sample: {sample.error}")
        return

    if sample.ms_scans is None:
        st.warning("No MS data available for this sample")
        return

    if sample.ms_times is None or len(sample.ms_times) == 0:
        st.warning("No MS time axis available for this sample")
        return

    mz_targets = st.session_state.mz_targets
    if not mz_targets:
        st.info("Add at least one target m/z value in sidebar settings.")
        return

    # Display options
    col1, col2 = st.columns(2)
    with col1:
        overlay = st.checkbox("Overlay EICs", value=True)
    with col2:
        normalize = st.checkbox("Normalize", value=True)

    st.divider()
    st.subheader("Peak Integration Selection")

    min_time = float(sample.ms_times[0])
    max_time = float(sample.ms_times[-1])

    selected_peak_windows: dict[str, list[dict]] = {}
    peak_data = []
    per_peak_data = []
    per_peak_data_by_mz: dict[str, list[dict]] = {}

    def rows_to_csv(rows: list[dict]) -> str:
        """Serialize list-of-dicts table rows to CSV text."""
        if not rows:
            return ""
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    for mz in mz_targets:
        eic = extract_eic(sample, mz, settings['mz_window'])
        if eic is None:
            continue

        smoothed = smooth_data(eic, settings['eic_smoothing'])
        total_area = calculate_peak_area(sample.ms_times, smoothed)
        auto_peaks = find_peaks(sample.ms_times, smoothed)

        # Stable key prefix per sample + target.
        sample_key = sample.name.replace(" ", "_").replace(".", "_")
        mz_key = f"{mz:.4f}".replace(".", "_")
        key_prefix = f"eic_batch_{sample_key}_{mz_key}"

        selected_rows = []
        selected_windows = []
        per_mz_rows = []
        peak_number = 1

        with st.expander(f"m/z {mz:.2f} peak windows", expanded=False):
            if auto_peaks:
                st.caption("Auto-detected peaks")
            else:
                st.caption("No auto peaks detected. Add manual windows below.")

            for i, peak in enumerate(auto_peaks):
                use_key = f"{key_prefix}_auto_use_{i}"
                start_key = f"{key_prefix}_auto_start_{i}"
                end_key = f"{key_prefix}_auto_end_{i}"

                if use_key not in st.session_state:
                    st.session_state[use_key] = True
                if start_key not in st.session_state:
                    st.session_state[start_key] = float(peak.get('start_time', peak['time']))
                if end_key not in st.session_state:
                    st.session_state[end_key] = float(peak.get('end_time', peak['time']))

                cols = st.columns([1.8, 1.6, 1.6, 1.6])
                with cols[0]:
                    use_peak = st.checkbox(
                        f"P{i + 1} auto ({float(peak['time']):.2f} min)",
                        key=use_key
                    )
                with cols[1]:
                    start_val = st.number_input(
                        "Start",
                        min_value=min_time,
                        max_value=max_time,
                        key=start_key,
                        format="%.3f",
                        label_visibility="collapsed"
                    )
                with cols[2]:
                    end_val = st.number_input(
                        "End",
                        min_value=min_time,
                        max_value=max_time,
                        key=end_key,
                        format="%.3f",
                        label_visibility="collapsed"
                    )

                start_val = float(start_val)
                end_val = float(end_val)
                row_area = calculate_peak_area(sample.ms_times, smoothed, start_val, end_val) if end_val > start_val else 0.0
                with cols[3]:
                    st.caption(f"Area: {row_area:.2e}")

                peak_row = {
                    'm/z': f"{mz:.2f}",
                    'Peak': f"P{i + 1}",
                    'Type': "Auto",
                    'Selected': "Yes" if use_peak else "No",
                    'Apex (min)': f"{float(peak['time']):.2f}",
                    'Start (min)': f"{start_val:.3f}",
                    'End (min)': f"{end_val:.3f}",
                    'Area': f"{row_area:.2e}",
                }
                per_peak_data.append(peak_row)
                per_mz_rows.append(peak_row)

                if use_peak and end_val > start_val:
                    selected_rows.append({
                        "label": f"P{peak_number}",
                        "apex_time": float(peak['time']),
                        "start": start_val,
                        "end": end_val,
                        "area": row_area
                    })
                    selected_windows.append({
                        "number": peak_number,
                        "apex_time": float(peak['time']),
                        "start": start_val,
                        "end": end_val
                    })
                    peak_number += 1

            st.caption("Manual peaks")
            manual_count_key = f"{key_prefix}_manual_count"
            if manual_count_key not in st.session_state:
                st.session_state[manual_count_key] = 0

            manual_count = int(st.number_input(
                "Additional manual peaks",
                min_value=0,
                max_value=10,
                step=1,
                key=manual_count_key
            ))

            for j in range(manual_count):
                manual_use_key = f"{key_prefix}_manual_use_{j}"
                manual_start_key = f"{key_prefix}_manual_start_{j}"
                manual_end_key = f"{key_prefix}_manual_end_{j}"

                if manual_use_key not in st.session_state:
                    st.session_state[manual_use_key] = True
                if manual_start_key not in st.session_state:
                    if auto_peaks:
                        center = float(auto_peaks[min(j, len(auto_peaks) - 1)]['time'])
                    else:
                        center = (min_time + max_time) / 2.0
                    st.session_state[manual_start_key] = max(min_time, center - 0.04)
                if manual_end_key not in st.session_state:
                    if auto_peaks:
                        center = float(auto_peaks[min(j, len(auto_peaks) - 1)]['time'])
                    else:
                        center = (min_time + max_time) / 2.0
                    st.session_state[manual_end_key] = min(max_time, center + 0.04)

                mcols = st.columns([1.8, 1.6, 1.6, 1.6])
                with mcols[0]:
                    use_peak = st.checkbox(
                        f"M{j + 1} manual",
                        key=manual_use_key
                    )
                with mcols[1]:
                    start_val = st.number_input(
                        "Start",
                        min_value=min_time,
                        max_value=max_time,
                        key=manual_start_key,
                        format="%.3f",
                        label_visibility="collapsed"
                    )
                with mcols[2]:
                    end_val = st.number_input(
                        "End",
                        min_value=min_time,
                        max_value=max_time,
                        key=manual_end_key,
                        format="%.3f",
                        label_visibility="collapsed"
                    )

                start_val = float(start_val)
                end_val = float(end_val)
                apex_time = (start_val + end_val) / 2.0
                row_area = calculate_peak_area(sample.ms_times, smoothed, start_val, end_val) if end_val > start_val else 0.0
                with mcols[3]:
                    st.caption(f"Area: {row_area:.2e}")

                peak_row = {
                    'm/z': f"{mz:.2f}",
                    'Peak': f"M{j + 1}",
                    'Type': "Manual",
                    'Selected': "Yes" if use_peak else "No",
                    'Apex (min)': f"{apex_time:.2f}",
                    'Start (min)': f"{start_val:.3f}",
                    'End (min)': f"{end_val:.3f}",
                    'Area': f"{row_area:.2e}",
                }
                per_peak_data.append(peak_row)
                per_mz_rows.append(peak_row)

                if use_peak and end_val > start_val:
                    selected_rows.append({
                        "label": f"P{peak_number}",
                        "apex_time": apex_time,
                        "start": start_val,
                        "end": end_val,
                        "area": row_area
                    })
                    selected_windows.append({
                        "number": peak_number,
                        "apex_time": apex_time,
                        "start": start_val,
                        "end": end_val
                    })
                    peak_number += 1

        selected_peak_windows[f"{mz:.4f}"] = selected_windows
        per_peak_data_by_mz[f"{mz:.2f}"] = per_mz_rows
        selected_area = sum(r["area"] for r in selected_rows)
        main_selected = max(selected_rows, key=lambda x: x["area"]) if selected_rows else None
        peak_data.append({
            'm/z': f"{mz:.2f}",
            'Total Area (full trace)': f"{total_area:.2e}",
            'Selected Peaks': len(selected_rows),
            'Selected Area': f"{selected_area:.2e}",
            'Main Selected Peak': (
                f"{main_selected['label']} @ {main_selected['apex_time']:.2f} min"
                if main_selected else "N/A"
            )
        })

    # Generate figure with selected integration windows.
    st.divider()
    fig = create_eic_comparison_figure(
        sample,
        mz_targets,
        mz_window=settings['mz_window'],
        smoothing=settings['eic_smoothing'],
        overlay=overlay,
        normalize=normalize,
        selected_peaks_by_mz=selected_peak_windows
    )
    st.pyplot(fig, use_container_width=True)

    # Export buttons directly under graph
    col1, col2, col3 = st.columns(3)
    with col1:
        png_data = export_figure(fig, dpi=settings['export_dpi'])
        st.download_button(
            label="Download PNG",
            data=png_data,
            file_name=f"{sample.name}_eic_batch.png",
            mime="image/png"
        )
    with col2:
        svg_data = export_figure_svg(fig)
        st.download_button(
            label="Download SVG",
            data=svg_data,
            file_name=f"{sample.name}_eic_batch.svg",
            mime="image/svg+xml"
        )
    with col3:
        pdf_data = export_figure_pdf(fig, dpi=settings['export_dpi'])
        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name=f"{sample.name}_eic_batch.pdf",
            mime="application/pdf"
        )

    # Peak areas table
    st.subheader("Peak Areas (Summary)")
    if peak_data:
        render_html_table(peak_data)
        summary_csv = rows_to_csv(peak_data)
        st.download_button(
            label="Download Summary CSV",
            data=summary_csv.encode("utf-8"),
            file_name=f"{sample.name}_eic_peak_summary.csv",
            mime="text/csv",
            key=f"eic_summary_csv_{sample.name}"
        )

    st.subheader("Peak Areas (Per Peak)")
    if per_peak_data:
        all_peaks_csv = rows_to_csv(per_peak_data)
        st.download_button(
            label="Download All Peaks CSV",
            data=all_peaks_csv.encode("utf-8"),
            file_name=f"{sample.name}_eic_peak_areas_all.csv",
            mime="text/csv",
            key=f"eic_all_peaks_csv_{sample.name}"
        )

        for mz_label, rows in per_peak_data_by_mz.items():
            st.markdown(f"**m/z {mz_label}**")
            render_html_table(rows, max_lines=20)
            mz_csv = rows_to_csv(rows)
            safe_mz = mz_label.replace(".", "_")
            st.download_button(
                label=f"Download CSV (m/z {mz_label})",
                data=mz_csv.encode("utf-8"),
                file_name=f"{sample.name}_eic_peak_areas_mz_{safe_mz}.csv",
                mime="text/csv",
                key=f"eic_peaks_csv_{sample.name}_{safe_mz}"
            )


def deconvolution_analysis(sample, settings):
    """Display protein deconvolution analysis view."""
    import matplotlib.pyplot as plt

    st.header(f"Protein Deconvolution: {sample.name}")

    if sample.error:
        st.error(f"Error loading sample: {sample.error}")
        return

    if sample.ms_scans is None or sample.ms_times is None:
        st.warning("No MS data available for this sample")
        return

    # Time range selection
    st.subheader("Select Time Region")

    min_time = float(sample.ms_times[0])
    max_time = float(sample.ms_times[-1])

    # Track which sample we're working with - reset values if sample changed
    if 'deconv_current_sample' not in st.session_state:
        st.session_state.deconv_current_sample = None
    if 'deconv_widget_key' not in st.session_state:
        st.session_state.deconv_widget_key = 0

    sample_changed = st.session_state.deconv_current_sample != sample.name
    if sample_changed:
        st.session_state.deconv_current_sample = sample.name
        st.session_state.deconv_widget_key += 1  # Force widget recreation
        # Reset time values to be re-initialized with auto-detect
        if 'deconv_start_val' in st.session_state:
            del st.session_state.deconv_start_val
        if 'deconv_end_val' in st.session_state:
            del st.session_state.deconv_end_val
        # Clear previous deconvolution output for the old sample
        if 'deconv_results' in st.session_state:
            del st.session_state.deconv_results
        if 'deconv_auto_profile_note' in st.session_state:
            del st.session_state.deconv_auto_profile_note
        if 'deconv_last_run_sig' in st.session_state:
            del st.session_state.deconv_last_run_sig

    # Auto-detect main peak
    if 'deconv_auto_start' not in st.session_state or sample_changed:
        st.session_state.deconv_auto_start = None
        st.session_state.deconv_auto_end = None
        st.session_state.deconv_autorun_pending = False
        st.session_state.deconv_last_autorun_sig = None

    if sample.tic is not None:
        auto_start, auto_end = _detect_deconvolution_window(sample)
        if auto_end > auto_start:
            st.session_state.deconv_auto_start = auto_start
            st.session_state.deconv_auto_end = auto_end

    # Initialize time values in session state - use auto-detected values if available
    if 'deconv_start_val' not in st.session_state:
        if st.session_state.deconv_auto_start is not None:
            st.session_state.deconv_start_val = st.session_state.deconv_auto_start
            st.session_state.deconv_autorun_pending = True
        else:
            st.session_state.deconv_start_val = min_time
    if 'deconv_end_val' not in st.session_state:
        if st.session_state.deconv_auto_end is not None:
            st.session_state.deconv_end_val = st.session_state.deconv_auto_end
            st.session_state.deconv_autorun_pending = True
        else:
            st.session_state.deconv_end_val = min(min_time + 1.0, max_time)

    # Auto-detect button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        auto_detect_clicked = st.button("Auto-detect main peak", type="secondary")
    with col2:
        if st.session_state.deconv_auto_start is not None:
            st.caption(f"Detected: {st.session_state.deconv_auto_start:.2f} - {st.session_state.deconv_auto_end:.2f} min")

    # Apply auto-detect values if button was clicked
    if auto_detect_clicked and st.session_state.deconv_auto_start is not None:
        st.session_state.deconv_start_val = st.session_state.deconv_auto_start
        st.session_state.deconv_end_val = st.session_state.deconv_auto_end
        st.session_state.deconv_widget_key += 1  # Force widget recreation
        st.session_state.deconv_autorun_pending = True
        st.rerun()

    # Helper to parse time input (accepts comma or period as decimal)
    def parse_time(text, default, min_val, max_val):
        try:
            # Replace comma with period for European format
            val = float(text.replace(',', '.'))
            return max(min_val, min(max_val, val))
        except (ValueError, AttributeError):
            return default

    # Format time without trailing zeros
    def format_time(val):
        formatted = f"{val:.3f}".rstrip('0').rstrip('.')
        return formatted if formatted else "0"

    # Text inputs for time (accepts comma as decimal separator)
    widget_key = st.session_state.deconv_widget_key
    col1, col2 = st.columns(2)
    with col1:
        start_text = st.text_input(
            "Start time (min)",
            value=format_time(st.session_state.deconv_start_val),
            key=f"deconv_start_{widget_key}"
        )
        start_time = parse_time(start_text, st.session_state.deconv_start_val, min_time, max_time)
    with col2:
        end_text = st.text_input(
            "End time (min)",
            value=format_time(st.session_state.deconv_end_val),
            key=f"deconv_end_{widget_key}"
        )
        end_time = parse_time(end_text, st.session_state.deconv_end_val, min_time, max_time)

    # Update session state from parsed values
    st.session_state.deconv_start_val = start_time
    st.session_state.deconv_end_val = end_time

    # Show TIC with region selector
    if sample.tic is not None:
        fig_tic, ax = plt.subplots(figsize=(12, 3))
        ax.plot(sample.ms_times, sample.tic, 'b-', linewidth=0.8)
        ax.axvspan(start_time, end_time, alpha=0.3, color='yellow', label='Selected region')
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("TIC Intensity")
        ax.set_title("TIC - Select region (adjust times above)")
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0), useMathText=True)
        _shift_sci_offset_left(ax)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_tic, use_container_width=True)
        plt.close(fig_tic)

    # Spectrum extraction — default to Average (sum) for best local machine agreement
    extraction_mode = "Average (sum)"
    apex_n_scans = 1

    # Always show mass spectrum for selected region
    st.subheader(f"Mass Spectrum ({format_time(start_time)} - {format_time(end_time)} min)")

    if extraction_mode == "Peak apex":
        mz, intensity = extract_apex_spectrum(sample, start_time, end_time, n_scans=apex_n_scans)
    else:
        mz, intensity = sum_spectra_in_range(sample, start_time, end_time)

    if len(mz) > 0:
        # m/z range filter (dynamic keys to update when time range changes)
        range_key = f"{start_time:.3f}_{end_time:.3f}"
        col1, col2 = st.columns(2)
        with col1:
            mz_min_display = st.number_input("Display m/z min", value=float(mz[0]), step=10.0, key=f"mz_min_{range_key}")
        with col2:
            mz_max_display = st.number_input("Display m/z max", value=float(mz[-1]), step=10.0, key=f"mz_max_{range_key}")

        # Filter for display
        display_mask = (mz >= mz_min_display) & (mz <= mz_max_display)
        mz_display = mz[display_mask]
        intensity_display = intensity[display_mask]

        # Find peaks for labeling (use centroiding for accurate m/z)
        peaks = find_spectrum_peaks(mz_display, intensity_display, height_threshold=0.05, min_distance=3, use_centroid=True)
        top_peaks = peaks[:15]  # Label top 15 peaks

        # Plot mass spectrum
        fig_ms, ax = plt.subplots(figsize=(14, 5))
        ax.plot(mz_display, intensity_display, 'b-', linewidth=0.8)

        # Add peak labels (whole numbers since data is ~1 Da resolution)
        for peak in top_peaks:
            ax.annotate(
                f"{peak['mz']:.0f}",
                xy=(peak['mz'], peak['intensity']),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=7,
                rotation=90
            )

        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Summed Mass Spectrum ({format_time(start_time)} - {format_time(end_time)} min)")
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0), useMathText=True)
        _shift_sci_offset_left(ax)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(mz_min_display, mz_max_display)
        # Add headroom for labels (20% extra at top)
        y_max = intensity_display.max() if len(intensity_display) > 0 else 1
        ax.set_ylim(0, y_max * 1.2)
        plt.tight_layout()
        st.pyplot(fig_ms, use_container_width=True)

        # Export buttons for summed mass spectrum
        col1, col2, col3 = st.columns(3)
        with col1:
            png_data = export_figure(fig_ms, dpi=settings['export_dpi'])
            st.download_button(
                label="Download PNG",
                data=png_data,
                file_name=f"{sample.name}_mass_spectrum.png",
                mime="image/png",
                key="ms_png"
            )
        with col2:
            svg_data = export_figure_svg(fig_ms)
            st.download_button(
                label="Download SVG",
                data=svg_data,
                file_name=f"{sample.name}_mass_spectrum.svg",
                mime="image/svg+xml",
                key="ms_svg"
            )
        with col3:
            pdf_data = export_figure_pdf(fig_ms, dpi=settings['export_dpi'])
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=f"{sample.name}_mass_spectrum.pdf",
                mime="application/pdf",
                key="ms_pdf"
            )
        plt.close(fig_ms)

        # Store spectrum in session state for deconvolution
        st.session_state.deconv_mz = mz
        st.session_state.deconv_intensity = intensity
        st.session_state.deconv_time_range = (start_time, end_time)
    else:
        st.warning("No mass spectrum data found in selected region")
        return

    st.divider()

    # Deconvolution parameters
    st.subheader("Deconvolution")

    # Deconvolution method — default to Local LC-MS machine-like for best accuracy
    method = "Local LC-MS machine-like"

    # Defaults for Local LC-MS machine-like parameters
    pwhh = 0.6
    mw_agreement_pct = 0.02           # Default: 0.02%
    noise_cutoff = 1000.0             # Default: 1000 counts
    abundance_cutoff_pct = 5.0        # Default: 5%
    mw_assign_cutoff_pct = 40.0       # Default: 40%
    envelope_cutoff_pct = 50.0        # Default: 50%
    contig_min = 3
    use_mz_agreement = False
    use_monoisotopic = False
    include_singly_charged = False
    min_peaks = 3
    max_peaks = 50
    mass_tolerance = 1.0
    low_mw = 500
    high_mw = 50000
    min_charge = 1
    max_charge = 50

    # Display settings (outside expander for easy access)
    top_n_masses = st.slider("Show top N masses", min_value=1, max_value=20, value=5, key="top_n_masses")
    show_obs_calc = True
    st.session_state["deconv_show_obs_calc"] = True
    calc_mass_da = None
    calc_mass_text = st.text_input(
        "Calculated mass (Da, optional)",
        value=st.session_state.get("deconv_calc_mass_text", ""),
        key="deconv_calc_mass_text",
        placeholder="e.g. 27848"
    )
    if calc_mass_text.strip():
        try:
            calc_mass_da = float(calc_mass_text.strip().replace(",", "."))
        except ValueError:
            st.warning("Calculated mass must be numeric (e.g., 27848).")

    with st.expander("Deconvolution Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            low_mw = st.number_input("Low MW (Da)", min_value=100, max_value=500000, value=500, step=100)
        with col2:
            high_mw = st.number_input("High MW (Da)", min_value=100, max_value=500000, value=50000, step=1000)

        col1, col2, col3 = st.columns(3)
        with col1:
            min_charge = st.number_input("Min charge", min_value=1, max_value=100, value=1)
        with col2:
            max_charge = st.number_input("Max charge", min_value=1, max_value=100, value=50)
        with col3:
            noise_cutoff = st.number_input("Noise cutoff (counts)", min_value=0.0, max_value=1e9, value=1000.0, step=100.0,
                                           help="Default: 1000")
        include_singly_charged = st.checkbox(
            "Include singly-charged (z=1) ions",
            value=False,
            help="Off by default to keep protein deconvolution ranking focused on charge envelopes."
        )

        # Expert mode — hidden by default
        expert_mode = st.checkbox("Expert mode", value=False, key="deconv_expert_mode")

        if expert_mode:
            st.caption("Method & extraction")
            col1, col2, col3 = st.columns(3)
            with col1:
                method = st.selectbox(
                    "Deconvolution method",
                    ["Local LC-MS machine-like", "Simple"],
                    index=0,
                    key="deconv_method"
                )
            with col2:
                extraction_mode = st.selectbox(
                    "Spectrum extraction",
                    ["Average (sum)", "Peak apex"],
                    index=0,
                    key="deconv_spectrum_mode"
                )
            with col3:
                if extraction_mode == "Peak apex":
                    apex_n_scans = st.selectbox(
                        "Apex window",
                        [1, 3, 5, 7],
                        index=1,
                        format_func=lambda x: f"{x} scan{'s' if x > 1 else ''}",
                        key="apex_n_scans",
                        help="Number of scans to average around the TIC apex."
                    )

            st.caption("Advanced parameters")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                min_peaks = st.number_input("Min peaks in set", min_value=2, max_value=50, value=3)
            with col2:
                max_peaks = st.number_input("Max peaks in set", min_value=2, max_value=100, value=50)
            with col3:
                mass_tolerance = st.number_input("Mass tolerance (Da)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            with col4:
                pwhh = st.number_input("Ion PWHH (Da)", min_value=0.05, max_value=5.0, value=0.6, step=0.05)

            if method == "Local LC-MS machine-like":
                col1, col2, col3 = st.columns(3)
                with col1:
                    mw_agreement_pct = st.number_input("MW agreement (%)", min_value=0.001, max_value=5.0, value=0.02, step=0.01,
                                                       help="Default: 0.02%")
                with col2:
                    abundance_cutoff_pct = st.number_input("Abundance cutoff (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0,
                                                           help="Default: 5%")
                with col3:
                    mw_assign_cutoff_pct = st.number_input("MW assign cutoff (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0,
                                                           help="Default: 40%")

                col1, col2, col3 = st.columns(3)
                with col1:
                    envelope_cutoff_pct = st.number_input("Envelope cutoff (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                                                          help="Default: 50%")
                with col2:
                    contig_min = st.number_input("Min contiguous charges", min_value=1, max_value=50, value=3, step=1)
                with col3:
                    use_mz_agreement = st.checkbox("Use m/z agreement", value=False)

                use_monoisotopic = st.checkbox(
                    "Monoisotopic H+",
                    value=False,
                    help="Use monoisotopic proton mass (1.007276) instead of average (1.00784). Try toggling if masses are ~1-2 Da off."
                )

    def run_deconvolution():
        with st.spinner("Running deconvolution..."):
            auto_profile_note = None
            if method == "Local LC-MS machine-like":
                results, auto_profile_note = _run_local_machine_auto_profile(
                    mz,
                    intensity,
                    min_charge=int(min_charge),
                    max_charge=int(max_charge),
                    min_peaks=int(min_peaks),
                    noise_cutoff=float(noise_cutoff),
                    abundance_cutoff_pct=float(abundance_cutoff_pct),
                    mw_agreement_pct=float(mw_agreement_pct),
                    mw_assign_cutoff_pct=float(mw_assign_cutoff_pct),
                    envelope_cutoff_pct=float(envelope_cutoff_pct),
                    pwhh=float(pwhh),
                    low_mw=float(low_mw),
                    high_mw=float(high_mw),
                    contig_min=int(contig_min),
                    use_mz_agreement=bool(use_mz_agreement),
                    use_monoisotopic=bool(use_monoisotopic),
                )
            else:
                results = deconvolute_protein(
                    mz, intensity,
                    min_charge=min_charge,
                    max_charge=max_charge,
                    mass_tolerance=mass_tolerance,
                    min_peaks=min_peaks,
                    max_peaks=max_peaks
                )

            results = [r for r in results if low_mw <= r['mass'] <= high_mw]

            # Optional merge of singly-charged (z=1) species.
            if method == "Local LC-MS machine-like" and include_singly_charged:
                singly = detect_singly_charged(
                    mz, intensity,
                    noise_cutoff=noise_cutoff,
                    min_intensity_pct=abundance_cutoff_pct,
                    low_mw=low_mw,
                    high_mw=min(high_mw, 2000.0),
                    pwhh=pwhh,
                    use_monoisotopic_proton=use_monoisotopic,
                )
                results.extend(singly)

            # Sort all results by intensity descending
            results.sort(key=lambda x: x['intensity'], reverse=True)

            st.session_state.deconv_results = results
            st.session_state.deconv_mw_range = (low_mw, high_mw)
            st.session_state.deconv_use_monoisotopic = use_monoisotopic
            st.session_state.deconv_auto_profile_note = auto_profile_note
            # Mark results as fresh for current time range/parameters.
            st.session_state.deconv_last_run_sig = autorun_sig

    autorun_sig = (
        config.APP_VERSION,
        sample.name,
        round(start_time, 4),
        round(end_time, 4),
        extraction_mode,
        apex_n_scans if extraction_mode == "Peak apex" else 1,
        method,
        low_mw,
        high_mw,
        min_charge,
        max_charge,
        min_peaks,
        max_peaks,
        mass_tolerance,
        # Local LC-MS machine-like params
        pwhh if method == "Local LC-MS machine-like" else None,
        mw_agreement_pct if method == "Local LC-MS machine-like" else None,
        noise_cutoff if method == "Local LC-MS machine-like" else None,
        abundance_cutoff_pct if method == "Local LC-MS machine-like" else None,
        mw_assign_cutoff_pct if method == "Local LC-MS machine-like" else None,
        envelope_cutoff_pct if method == "Local LC-MS machine-like" else None,
        contig_min if method == "Local LC-MS machine-like" else None,
        use_mz_agreement if method == "Local LC-MS machine-like" else None,
        use_monoisotopic if method == "Local LC-MS machine-like" else None,
        include_singly_charged if method == "Local LC-MS machine-like" else None
    )
    if st.session_state.get('deconv_autorun_pending') and st.session_state.get('deconv_last_autorun_sig') != autorun_sig:
        run_deconvolution()
        st.session_state.deconv_last_autorun_sig = autorun_sig
        st.session_state.deconv_autorun_pending = False

    # Run deconvolution button
    if st.button("Run Deconvolution", type="primary"):
        run_deconvolution()

    results_are_current = st.session_state.get('deconv_last_run_sig') == autorun_sig
    if hasattr(st.session_state, 'deconv_results') and st.session_state.deconv_results and not results_are_current:
        st.info("Time range or parameters changed. Click Run Deconvolution to refresh results.")

    # Display results if available
    if hasattr(st.session_state, 'deconv_results') and st.session_state.deconv_results and results_are_current:
        results = st.session_state.deconv_results
        mz = st.session_state.deconv_mz
        intensity = st.session_state.deconv_intensity
        time_range = st.session_state.deconv_time_range

        # Build info caption
        caption_parts = []
        auto_profile_note = st.session_state.get('deconv_auto_profile_note')
        if auto_profile_note and not expert_mode:
            caption_parts.append(auto_profile_note)
        if expert_mode:
            caption_parts.append(f"Method: {method}")
            apex_info = f" ({apex_n_scans} scans)" if extraction_mode == "Peak apex" else ""
            caption_parts.append(f"Spectrum: {extraction_mode}{apex_info}")
            if method == "Local LC-MS machine-like" and use_monoisotopic:
                caption_parts.append("H+=1.007276 (mono)")
            if method == "Local LC-MS machine-like" and include_singly_charged:
                caption_parts.append("z=1 included")
        if caption_parts:
            st.caption(" | ".join(caption_parts))

        show_full_precision = st.checkbox("Show full precision masses", value=False, key="deconv_full_precision")

        st.divider()

        significant_results = _filter_display_deconvolution_results(
            results,
            expert_mode=expert_mode,
            min_rel_intensity=0.05,
        )
        if not expert_mode:
            st.subheader(f"Results ({len(significant_results)} components, ≥5% rel. intensity)")
        else:
            st.subheader(f"Results ({len(significant_results)} masses detected)")

        # Limit results to top N
        display_results = significant_results[:top_n_masses]

        # Results table
        result_data = []
        for i, r in enumerate(display_results):
            mass_val = r['mass'] if show_full_precision else f"{r['mass']:.2f}"
            std_val = r['mass_std'] if show_full_precision else f"{r['mass_std']:.2f}"
            result_data.append({
                'Rank': i + 1,
                'Mass (Da)': mass_val,
                'Std Dev': std_val,
                'Charge States': f"{min(r['charge_states'])}-{max(r['charge_states'])}",
                'Num Charges': r['num_charges'],
                'Rel. Intensity': f"{r['intensity'] / results[0]['intensity'] * 100:.1f}%"
            })

        render_text_table(result_data, list(result_data[0].keys()) if result_data else [])

        # Create figure
        style = {
            'fig_width': settings['fig_width'],
            'line_width': settings['line_width'],
            'show_grid': settings['show_grid'],
            'top_n_masses': top_n_masses,
            'deconv_x_min_da': settings['deconv_x_min_da'],
            'deconv_x_max_da': settings['deconv_x_max_da'],
            'deconv_show_title': settings['deconv_show_title'],
            'deconv_show_subtitle': settings['deconv_show_subtitle'],
            'deconv_show_obs_calc': show_obs_calc,
            'deconv_calc_mass_da': calc_mass_da,
        }

        fig = create_deconvolution_figure(sample, time_range[0], time_range[1], display_results, style)
        st.pyplot(fig, use_container_width=True)

        # Export buttons for 3-panel figure (placed directly under the graph)
        col1, col2, col3 = st.columns(3)
        with col1:
            png_data = export_figure(fig, dpi=settings['export_dpi'])
            st.download_button(
                label="Download PNG",
                data=png_data,
                file_name=f"{sample.name}_deconvolution.png",
                mime="image/png"
            )
        with col2:
            svg_data = export_figure_svg(fig)
            st.download_button(
                label="Download SVG",
                data=svg_data,
                file_name=f"{sample.name}_deconvolution.svg",
                mime="image/svg+xml"
            )
        with col3:
            pdf_data = export_figure_pdf(fig, dpi=settings['export_dpi'])
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=f"{sample.name}_deconvolution.pdf",
                mime="application/pdf"
            )

        # Standalone deconvoluted masses panel (same figure size as the 3-panel plot)
        st.subheader("Deconvoluted Masses (Standalone)")
        fig_deconv_only = create_deconvoluted_masses_figure(sample.name, display_results, style)
        st.pyplot(fig_deconv_only, use_container_width=True)

        export_key_base = sample.name.replace(" ", "_").replace(".", "_")
        col1, col2, col3 = st.columns(3)
        with col1:
            png_data_only = export_figure(fig_deconv_only, dpi=settings['export_dpi'])
            st.download_button(
                label="Download PNG (deconvoluted)",
                data=png_data_only,
                file_name=f"{sample.name}_deconvoluted_masses.png",
                mime="image/png",
                key=f"deconv_only_png_{export_key_base}"
            )
        with col2:
            svg_data_only = export_figure_svg(fig_deconv_only)
            st.download_button(
                label="Download SVG (deconvoluted)",
                data=svg_data_only,
                file_name=f"{sample.name}_deconvoluted_masses.svg",
                mime="image/svg+xml",
                key=f"deconv_only_svg_{export_key_base}"
            )
        with col3:
            pdf_data_only = export_figure_pdf(fig_deconv_only, dpi=settings['export_dpi'])
            st.download_button(
                label="Download PDF (deconvoluted)",
                data=pdf_data_only,
                file_name=f"{sample.name}_deconvoluted_masses.pdf",
                mime="application/pdf",
                key=f"deconv_only_pdf_{export_key_base}"
            )

        # Ion selection figure — shows which m/z peaks were used per component
        st.subheader("Ion Selection per Component")
        fig_ions = create_ion_selection_figure(mz, intensity, display_results, style)
        st.pyplot(fig_ions, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            png_ions = export_figure(fig_ions, dpi=settings['export_dpi'])
            st.download_button(
                label="Download PNG (ion selection)",
                data=png_ions,
                file_name=f"{sample.name}_ion_selection.png",
                mime="image/png",
                key=f"ion_sel_png_{export_key_base}"
            )
        with col2:
            svg_ions = export_figure_svg(fig_ions)
            st.download_button(
                label="Download SVG (ion selection)",
                data=svg_ions,
                file_name=f"{sample.name}_ion_selection.svg",
                mime="image/svg+xml",
                key=f"ion_sel_svg_{export_key_base}"
            )
        with col3:
            pdf_ions = export_figure_pdf(fig_ions, dpi=settings['export_dpi'])
            st.download_button(
                label="Download PDF (ion selection)",
                data=pdf_ions,
                file_name=f"{sample.name}_ion_selection.pdf",
                mime="application/pdf",
                key=f"ion_sel_pdf_{export_key_base}"
            )

        # Show theoretical vs observed m/z for selected mass
        st.subheader("Charge State Details")
        theo_list = significant_results[:10]
        selected_mass_idx = st.selectbox(
            "Select component",
            range(len(theo_list)),
            format_func=lambda i: f"{theo_list[i]['mass']:.2f} Da ({theo_list[i]['num_charges']} charges)",
            key="theo_mz_mass_select"
        )

        if selected_mass_idx is not None:
            selected_result = theo_list[selected_mass_idx]
            use_mono = st.session_state.get('deconv_use_monoisotopic', False)
            theoretical = get_theoretical_mz(
                selected_result['mass'],
                selected_result['charge_states'],
                use_monoisotopic_proton=use_mono
            )

            # Build observed lookup from ion data
            obs_by_charge = {}
            ion_mzs = selected_result.get('ion_mzs', [])
            ion_charges = selected_result.get('ion_charges', [])
            ion_ints = selected_result.get('ion_intensities', [])
            for mz_val, z, intensity_val in zip(ion_mzs, ion_charges, ion_ints):
                obs_by_charge[z] = (mz_val, intensity_val)

            max_ion_int = max(ion_ints) if ion_ints else 1.0

            theo_data = []
            for t in theoretical:
                z = t['charge']
                row = {
                    'Charge': f"+{z}",
                    'Theoretical m/z': f"{t['mz']:.4f}",
                }
                obs = obs_by_charge.get(z)
                if obs:
                    obs_mz, obs_int = obs
                    delta = obs_mz - t['mz']
                    rel_pct = obs_int / max_ion_int * 100
                    row['Observed m/z'] = f"{obs_mz:.4f}"
                    row['Delta (Da)'] = f"{delta:+.4f}"
                    row['Intensity'] = f"{obs_int:,.0f}"
                    row['Rel %'] = f"{rel_pct:.1f}%"
                else:
                    row['Observed m/z'] = "—"
                    row['Delta (Da)'] = "—"
                    row['Intensity'] = "—"
                    row['Rel %'] = "—"
                theo_data.append(row)
            render_html_table(theo_data)

    elif hasattr(st.session_state, 'deconv_results') and results_are_current:
        st.info("No protein masses detected. Try adjusting the parameters or selecting a different time region.")


def batch_deconvolution_analysis(samples: list, settings):
    """Run default deconvolution for multiple samples and show standalone panels."""
    st.header("Batch Deconvolution")

    valid_samples = []
    for sample in samples:
        if sample.error:
            st.warning(f"Skipping {sample.name}: {sample.error}")
            continue
        if sample.ms_scans is None or sample.ms_times is None:
            st.warning(f"Skipping {sample.name}: no MS data available")
            continue
        valid_samples.append(sample)

    if not valid_samples:
        st.info("No valid samples available for batch deconvolution.")
        return

    top_n_masses = st.slider(
        "Show top N masses",
        min_value=1,
        max_value=20,
        value=5,
        key="batch_deconv_top_n"
    )

    style = {
        'fig_width': settings['fig_width'],
        'line_width': settings['line_width'],
        'show_grid': settings['show_grid'],
        'top_n_masses': top_n_masses,
        'deconv_x_min_da': settings['deconv_x_min_da'],
        'deconv_x_max_da': settings['deconv_x_max_da'],
        'deconv_show_title': settings['deconv_show_title'],
        'deconv_show_subtitle': settings['deconv_show_subtitle'],
        'deconv_show_obs_calc': True,
        'deconv_calc_mass_da': None,
    }

    batch_calc_mass_text = st.session_state.get("deconv_calc_mass_text", "")
    if isinstance(batch_calc_mass_text, str) and batch_calc_mass_text.strip():
        try:
            style['deconv_calc_mass_da'] = float(batch_calc_mass_text.strip().replace(",", "."))
        except ValueError:
            style['deconv_calc_mass_da'] = None

    for idx, sample in enumerate(valid_samples):
        sample_title = sample.name[:-2] if sample.name.lower().endswith(".d") else sample.name
        st.subheader(f"{idx + 1}. {sample_title}")

        start_time, end_time = _detect_deconvolution_window(sample)
        st.caption(f"Auto window: {start_time:.2f} - {end_time:.2f} min")

        mz, intensity = sum_spectra_in_range(sample, start_time, end_time)
        if len(mz) == 0:
            st.warning("No mass spectrum data found in the selected region.")
            continue

        results = _run_default_deconvolution(mz, intensity)
        if not results:
            st.info("No masses detected for this sample.")
            continue

        results = _filter_display_deconvolution_results(
            results,
            expert_mode=False,
            min_rel_intensity=0.05,
        )

        display_results = results[:top_n_masses]
        fig = create_deconvoluted_masses_figure(sample.name, display_results, style)
        st.pyplot(fig, use_container_width=True)

        export_key_base = f"{idx}_{sample.name}".replace(" ", "_").replace(".", "_")
        col1, col2, col3 = st.columns(3)
        with col1:
            png_data = export_figure(fig, dpi=settings['export_dpi'])
            st.download_button(
                label="Download PNG",
                data=png_data,
                file_name=f"{sample.name}_batch_deconvoluted_masses.png",
                mime="image/png",
                key=f"batch_deconv_png_{export_key_base}"
            )
        with col2:
            svg_data = export_figure_svg(fig)
            st.download_button(
                label="Download SVG",
                data=svg_data,
                file_name=f"{sample.name}_batch_deconvoluted_masses.svg",
                mime="image/svg+xml",
                key=f"batch_deconv_svg_{export_key_base}"
            )
        with col3:
            pdf_data = export_figure_pdf(fig, dpi=settings['export_dpi'])
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=f"{sample.name}_batch_deconvoluted_masses.pdf",
                mime="application/pdf",
                key=f"batch_deconv_pdf_{export_key_base}"
            )

        plt.close(fig)
        if idx < len(valid_samples) - 1:
            st.divider()


def time_change_mass_spectra_analysis(samples: list, settings):
    """Show summed mass spectrum plots for 2+ samples."""
    st.header("Time Change - Summed Mass Spectra")

    if len(samples) < 2:
        st.warning("Select at least 2 samples to enable this view.")
        return

    valid_samples = []
    for sample in samples:
        if sample.error:
            st.warning(f"Skipping {sample.name}: {sample.error}")
            continue
        if sample.ms_scans is None or sample.ms_times is None:
            st.warning(f"Skipping {sample.name}: no MS data available")
            continue
        valid_samples.append(sample)

    if len(valid_samples) < 2:
        st.info("At least two valid MS samples are required.")
        return

    # Assignment + labels follow Time Progression behavior.
    if len(valid_samples) == 2:
        roles = ["Initial (t=0)", "Overnight"]
    else:
        roles = ["Initial (t=0)", "Mid timepoint", "Overnight"]
        for i in range(3, len(valid_samples)):
            roles.append(f"Sample {i + 1}")
    sample_names = [s.name for s in valid_samples]

    st.subheader("Sample Assignment")
    selected_indices: list[int] = []
    assign_cols = st.columns(len(roles))
    for i, (col, role) in enumerate(zip(assign_cols, roles)):
        with col:
            default_idx = min(i, len(sample_names) - 1)
            selected_name = st.selectbox(
                f"{role} sample",
                sample_names,
                index=default_idx,
                key=f"time_ms_role_{i}"
            )
            selected_indices.append(sample_names.index(selected_name))

    if len(set(selected_indices)) != len(selected_indices):
        st.warning("Choose a different sample for each role.")
        return

    ordered_samples = [valid_samples[i] for i in selected_indices]

    st.subheader("Sample Labels")
    labels = []
    label_cols = st.columns(len(ordered_samples))
    for i, (col, sample, role) in enumerate(zip(label_cols, ordered_samples, roles)):
        with col:
            label = st.text_input(
                f"Label for {sample.name}",
                value=role,
                key=f"label_time_ms_{i}"
            )
            labels.append(label)

    normalize = st.checkbox("Normalize each spectrum to 100%", value=True, key="time_ms_normalize")

    spectra = []
    window_rows = []
    for sample, label in zip(ordered_samples, labels):
        start_time, end_time = _detect_deconvolution_window(sample)
        mz, intensity = sum_spectra_in_range(sample, start_time, end_time)
        if len(mz) == 0:
            st.warning(f"{sample.name}: no mass spectrum data in auto-detected window.")
            continue

        plot_intensity = intensity.astype(float)
        if normalize and np.max(plot_intensity) > 0:
            plot_intensity = plot_intensity / np.max(plot_intensity) * 100.0

        spectra.append({
            "sample_name": sample.name,
            "label": label,
            "mz": mz,
            "intensity": plot_intensity,
            "start_time": start_time,
            "end_time": end_time
        })
        window_rows.append({
            "Label": label,
            "Sample": sample.name,
            "Window (min)": f"{start_time:.2f} - {end_time:.2f}"
        })

    if len(spectra) < 2:
        st.info("At least two spectra with valid MS data are required for overlay.")
        return

    st.divider()
    st.subheader("Overlayed Summed Mass Spectrum")
    render_text_table(window_rows, list(window_rows[0].keys()) if window_rows else [])

    # Match Time Progression color behavior.
    color_initial = settings['colors']['initial']
    color_mid = settings['colors']['mid']
    color_final = settings['colors']['final']
    if len(spectra) == 2:
        colors = [color_initial, color_final]
    else:
        colors = [color_initial, color_mid, color_final]
        for c in config.EIC_COLORS:
            if len(colors) >= len(spectra):
                break
            if c not in colors:
                colors.append(c)
        while len(colors) < len(spectra):
            colors.append(config.EIC_COLORS[len(colors) % len(config.EIC_COLORS)])

    fig_height = max(4.5, settings['fig_height_per_panel'] * 2.2)
    fig, ax = plt.subplots(figsize=(settings['fig_width'], fig_height))

    for i, spec in enumerate(spectra):
        color = colors[i % len(colors)]
        ax.plot(
            spec["mz"],
            spec["intensity"],
            linewidth=settings['line_width'],
            color=color,
            label=spec["label"]
        )

        peaks = find_spectrum_peaks(
            spec["mz"],
            spec["intensity"],
            height_threshold=0.10,
            min_distance=5,
            use_centroid=True
        )
        for peak in peaks:
            ax.annotate(
                f"{peak['mz']:.1f}",
                xy=(peak['mz'], peak['intensity']),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center',
                fontsize=6,
                color=color,
                rotation=90
            )

    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative Intensity (%)" if normalize else "Intensity")
    ax.set_title("Summed Mass Spectrum", fontweight='bold')
    ax.legend(loc='upper right')
    if settings['show_grid']:
        ax.grid(True, alpha=0.3)
    if settings['y_scale'] == 'log':
        ax.set_yscale('log')
    else:
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0), useMathText=True)
        _shift_sci_offset_left(ax)

    if settings['y_scale'] != 'log':
        y_max = max(np.max(spec["intensity"]) for spec in spectra)
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.15)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        png_data = export_figure(fig, dpi=settings['export_dpi'])
        st.download_button(
            label="Download PNG",
            data=png_data,
            file_name="time_change_summed_mass_spectrum_overlay.png",
            mime="image/png",
            key="time_ms_overlay_png"
        )
    with col2:
        svg_data = export_figure_svg(fig)
        st.download_button(
            label="Download SVG",
            data=svg_data,
            file_name="time_change_summed_mass_spectrum_overlay.svg",
            mime="image/svg+xml",
            key="time_ms_overlay_svg"
        )
    with col3:
        pdf_data = export_figure_pdf(fig, dpi=settings['export_dpi'])
        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name="time_change_summed_mass_spectrum_overlay.pdf",
            mime="application/pdf",
            key="time_ms_overlay_pdf"
        )

    plt.close(fig)

    # Additional offset view: keep original overlay unchanged and add a
    # diagonally shifted version to improve visual trace separation.
    st.divider()
    st.subheader("Overlayed Summed Mass Spectrum (Diagonal Offset View)")

    x_offset_step = 20.0  # m/z shift per overlaid trace
    if normalize:
        y_offset_step = 10.0  # relative-intensity units (%)
        y_offset_label = f"+{y_offset_step:.0f} intensity units"
    else:
        # Keep visible vertical separation in absolute-intensity mode.
        global_y_max = max(np.max(spec["intensity"]) for spec in spectra)
        y_offset_step = (0.10 * global_y_max) if global_y_max > 0 else 1.0
        y_offset_label = f"+{y_offset_step:.2g} intensity units"

    fig_offset, ax_offset = plt.subplots(figsize=(settings['fig_width'], fig_height))
    all_x_min = float("inf")
    all_x_max = float("-inf")
    all_y_max = float("-inf")

    for i, spec in enumerate(spectra):
        color = colors[i % len(colors)]
        x_offset = i * x_offset_step
        y_offset = i * y_offset_step

        mz_shifted = spec["mz"] + x_offset
        intensity_shifted = spec["intensity"] + y_offset

        all_x_min = min(all_x_min, float(np.min(mz_shifted)))
        all_x_max = max(all_x_max, float(np.max(mz_shifted)))
        all_y_max = max(all_y_max, float(np.max(intensity_shifted)))

        ax_offset.plot(
            mz_shifted,
            intensity_shifted,
            linewidth=settings['line_width'],
            color=color,
            label=spec["label"]
        )

    ax_offset.set_xlabel("m/z")
    ax_offset.set_ylabel("Relative Intensity (%)" if normalize else "Intensity")
    ax_offset.set_title(
        f"Summed Mass Spectrum (Diagonal Offset: +{x_offset_step:.0f} m/z, {y_offset_label} per trace)",
        fontweight='bold'
    )
    ax_offset.legend(loc='upper right')
    if settings['show_grid']:
        ax_offset.grid(True, alpha=0.3)

    # Offset view is intentionally linear so the additive shift is interpretable.
    ax_offset.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0), useMathText=True)
    _shift_sci_offset_left(ax_offset)

    if all_x_min < all_x_max:
        ax_offset.set_xlim(all_x_min, all_x_max)
    if all_y_max > 0:
        ax_offset.set_ylim(0, all_y_max * 1.10)

    plt.tight_layout()
    st.pyplot(fig_offset, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        png_data = export_figure(fig_offset, dpi=settings['export_dpi'])
        st.download_button(
            label="Download Offset PNG",
            data=png_data,
            file_name="time_change_summed_mass_spectrum_offset_overlay.png",
            mime="image/png",
            key="time_ms_offset_overlay_png"
        )
    with col2:
        svg_data = export_figure_svg(fig_offset)
        st.download_button(
            label="Download Offset SVG",
            data=svg_data,
            file_name="time_change_summed_mass_spectrum_offset_overlay.svg",
            mime="image/svg+xml",
            key="time_ms_offset_overlay_svg"
        )
    with col3:
        pdf_data = export_figure_pdf(fig_offset, dpi=settings['export_dpi'])
        st.download_button(
            label="Download Offset PDF",
            data=pdf_data,
            file_name="time_change_summed_mass_spectrum_offset_overlay.pdf",
            mime="application/pdf",
            key="time_ms_offset_overlay_pdf"
        )

    plt.close(fig_offset)


# ---------------------------------------------------------------------------
# Average amino acid residue masses (Da) for theoretical protein mass calc
# ---------------------------------------------------------------------------
AA_MASSES = {
    'G': 57.0519, 'A': 71.0788, 'V': 99.1326, 'L': 113.1594, 'I': 113.1594,
    'P': 97.1167, 'F': 147.1766, 'W': 186.2132, 'M': 131.1926, 'S': 87.0782,
    'T': 101.1051, 'C': 103.1388, 'Y': 163.1760, 'H': 137.1411, 'D': 115.0886,
    'E': 129.1155, 'N': 114.1038, 'Q': 128.1307, 'K': 128.1741, 'R': 156.1875,
}
WATER_MASS = 18.01524

# Common post-translational / adduct mass shifts
KNOWN_MODS = {
    'Oxidation (+O)': 15.999,
    'Acetylation': 42.011,
    'Phosphorylation': 79.966,
    'Methylation': 14.016,
    'Met loss (-M)': -131.040,
    'Met loss + Acetyl': -89.030,
    'Atto488': 572.0,
    'Ubiquitin GG': 114.043,
    'Disulfide (-2H)': -2.016,
    'Deamidation': 0.984,
    'Na adduct': 21.982,
    'K adduct': 37.956,
    'Glucuronidation': 176.032,
    'Formic acid adduct': 46.005,
    'TFA adduct': 113.993,
    'Unknown mod (×1)': 251.30,
    'Unknown mod (×2)': 502.60,
    'Unknown mod (×3)': 753.90,
}


def mass_calculator_tab(sample_list: list, settings):
    """Mass calculator: compare theoretical vs observed deconvolution masses."""
    st.header("Mass Calculator")

    theoretical_masses = []  # list of floats

    user_input = st.text_input("Amino acid sequence or mass(es) (Da), comma-separated",
                              key="masscalc_input",
                              placeholder="MQIFVKTLTGKTITLEVEPS…  or  10651.3  or  3497.5, 3748.6, 3999.8")

    if user_input and user_input.strip():
        stripped = user_input.strip()
        # Try parsing as comma-separated numbers first
        parts = [p.strip() for p in stripped.split(",") if p.strip()]
        parsed_numbers = []
        for p in parts:
            try:
                v = float(p)
                if v > 0:
                    parsed_numbers.append(v)
            except ValueError:
                parsed_numbers = []
                break
        if parsed_numbers:
            theoretical_masses = parsed_numbers[:10]  # max 10
            labels = ", ".join(f"{m:.2f}" for m in theoretical_masses)
            st.metric("Input mass(es)", f"{labels} Da")
        else:
            # Treat as amino acid sequence
            seq = ''.join(ch.upper() for ch in stripped if ch.isalpha())
            if seq:
                unknown = sorted({ch for ch in seq if ch not in AA_MASSES})
                if unknown:
                    st.warning(f"Unknown residue(s): {', '.join(unknown)} — these are ignored.")
                mass = sum(AA_MASSES.get(ch, 0.0) for ch in seq) + WATER_MASS
                theoretical_masses = [mass]
                st.metric("Theoretical average mass", f"{mass:.2f} Da")
                st.caption(f"Sequence length: {len(seq)} residues")

    if not theoretical_masses:
        st.info("Enter an amino acid sequence or mass(es) in Da above to begin.")
        return

    # Primary theoretical mass (first one) used for modification reference
    theoretical_mass = theoretical_masses[0]

    st.divider()

    # Custom modifications
    st.subheader("Modifications")
    custom_mods = dict(KNOWN_MODS)

    with st.expander("Add custom modification"):
        cmod_name = st.text_input("Name", key="masscalc_cmod_name", placeholder="e.g. PEG-5k")
        cmod_mass = st.number_input("Mass shift (Da)", value=0.0, step=0.01, format="%.3f",
                                    key="masscalc_cmod_mass")
        if cmod_name.strip() and cmod_mass != 0:
            custom_mods[cmod_name.strip()] = cmod_mass

    # Tolerance for matching
    tol = st.slider("Matching tolerance (Da)", min_value=0.5, max_value=10.0, value=2.0,
                    step=0.5, key="masscalc_tol")

    # Show modification reference table (both +mod and -mod directions)
    mod_ref = []
    for k, v in custom_mods.items():
        mod_ref.append({'Modification': f"+{k}" if v >= 0 else k,
                        'Δm (Da)': f"{v:+.3f}",
                        'Expected mass': f"{theoretical_mass + v:.2f}"})
        mod_ref.append({'Modification': f"−{k}" if v >= 0 else f"+{k.lstrip('-')}",
                        'Δm (Da)': f"{-v:+.3f}",
                        'Expected mass': f"{theoretical_mass - v:.2f}"})
    st.caption("Known modifications relative to theoretical mass (both directions):")
    render_text_table(mod_ref, ['Modification', 'Δm (Da)', 'Expected mass'])

    # Compare against deconv results
    st.divider()
    st.subheader("Comparison with Deconvolution Results")

    # Sample selector
    if len(sample_list) == 1:
        sample = sample_list[0]
    else:
        selected_idx = st.selectbox(
            "Select sample",
            range(len(sample_list)),
            format_func=lambda i: sample_list[i].name,
            key="masscalc_sample_idx"
        )
        sample = sample_list[selected_idx]

    if sample.error or sample.ms_scans is None:
        st.warning(f"No MS data available for {sample.name}.")
        return

    # Try cached results first, then auto-deconvolute
    deconv_results = None
    cached_results = st.session_state.get('deconv_results')
    cached_sample = st.session_state.get('deconv_current_sample')
    if cached_results and cached_sample == sample.name:
        deconv_results = cached_results
    else:
        # Auto-run deconvolution for this sample
        if sample.tic is not None:
            auto_start, auto_end = _detect_deconvolution_window(sample)
            if auto_end > auto_start:
                mz, intensity = sum_spectra_in_range(sample, auto_start, auto_end)
                if mz is not None and len(mz) > 0:
                    with st.spinner(f"Running auto-deconvolution for {sample.name}..."):
                        deconv_results = _run_default_deconvolution(mz, intensity)

    if not deconv_results:
        st.info(f"Could not obtain deconvolution results for {sample.name}. "
                "Try running deconvolution manually in the Deconvolution tab.")
        return

    st.caption(f"Comparing against deconvolution results for **{sample.name}**")

    top_n = st.slider("Number of components to show", min_value=1, max_value=min(20, len(deconv_results)),
                       value=min(5, len(deconv_results)), step=1, key="masscalc_top_n")

    # Same color palette as the deconvoluted masses figure
    rank_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Build header: if multiple input masses, show "Ref. Mass" column
    multi = len(theoretical_masses) > 1
    columns = ['Rank', 'Observed (Da)']
    if multi:
        columns.append('Ref. Mass')
    columns += ['Δm (Da)', 'Rel. Int.', 'Match']
    header_html = "".join(
        f'<th style="text-align:left; padding:0.45rem 0.6rem; border-bottom:2px solid #666;">{escape(c)}</th>'
        for c in columns
    )

    body_rows = []
    n_matches = 0
    for i, r in enumerate(deconv_results[:top_n]):
        obs = r['mass']
        rel_int = r['intensity'] / deconv_results[0]['intensity'] * 100
        rank_color = rank_colors[i % len(rank_colors)]

        # Find closest theoretical mass for delta calculation
        best_ref = min(theoretical_masses, key=lambda m: abs(obs - m))
        delta = obs - best_ref

        # Check annotations against each theoretical mass
        annotations = []
        for tm in theoretical_masses:
            d = obs - tm
            if abs(d) <= tol:
                annotations.append("Observed")
            for mod_name, mod_mass in custom_mods.items():
                # Match both +mod and -mod (modified or unmodified direction)
                if abs(d - mod_mass) <= tol:
                    annotations.append(f"+{mod_name}" if mod_mass >= 0 else mod_name)
                if abs(d + mod_mass) <= tol:
                    sign = "−" if mod_mass >= 0 else "+"
                    annotations.append(f"{sign}{mod_name}")
        # Deduplicate while preserving order
        seen = set()
        unique_annotations = []
        for a in annotations:
            if a not in seen:
                seen.add(a)
                unique_annotations.append(a)

        is_match = len(unique_annotations) > 0
        annotation_str = ", ".join(unique_annotations) if unique_annotations else "\u2014"
        vals = [str(i + 1), f"{obs:.2f}"]
        if multi:
            vals.append(f"{best_ref:.2f}")
        vals += [f"{delta:+.2f}", f"{rel_int:.1f}%", annotation_str]

        if is_match:
            n_matches += 1
            color = rank_color
            weight = "bold"
        else:
            color = "#666"
            weight = "normal"

        cells = "".join(
            f'<td style="padding:0.40rem 0.6rem; border-bottom:1px solid #3f3f3f; '
            f'font-weight:{weight}; color:{color};">{escape(v)}</td>'
            for v in vals
        )
        body_rows.append(f'<tr>{cells}</tr>')

    table_html = (
        '<table style="width:100%; border-collapse:collapse; font-size:0.92rem;">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>'
    )
    st.markdown(f'<div style="overflow-x:auto; padding:0;">{table_html}</div>',
                unsafe_allow_html=True)

    mass_str = ", ".join(f"{m:.2f}" for m in theoretical_masses)
    if n_matches:
        st.success(f"Found {n_matches} matching component(s) within {tol:.1f} Da tolerance.")
    else:
        st.warning(f"No components found within {tol:.1f} Da of {mass_str} Da.")

    # Deconvoluted masses figure with theoretical mass reference line
    st.divider()
    display_results = deconv_results[:top_n]
    fig_style = {
        'fig_width': settings['fig_width'],
        'show_grid': False,
        'deconv_x_min_da': settings['deconv_x_min_da'],
        'deconv_x_max_da': settings['deconv_x_max_da'],
        'deconv_show_title': settings['deconv_show_title'],
        'deconv_show_subtitle': settings['deconv_show_subtitle'],
        'deconv_show_obs_calc': True,
        'deconv_calc_mass_da': theoretical_masses if len(theoretical_masses) > 1 else theoretical_mass,
    }
    fig_calc = create_deconvoluted_masses_figure(sample.name, display_results, fig_style)
    st.pyplot(fig_calc, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    export_key = sample.name.replace(" ", "_").replace(".", "_")
    with col1:
        png_data = export_figure(fig_calc, dpi=settings['export_dpi'])
        st.download_button("Download PNG", data=png_data,
                           file_name=f"{sample.name}_masscalc.png", mime="image/png",
                           key=f"masscalc_png_{export_key}")
    with col2:
        svg_data = export_figure_svg(fig_calc)
        st.download_button("Download SVG", data=svg_data,
                           file_name=f"{sample.name}_masscalc.svg", mime="image/svg+xml",
                           key=f"masscalc_svg_{export_key}")
    with col3:
        pdf_data = export_figure_pdf(fig_calc, dpi=settings['export_dpi'])
        st.download_button("Download PDF", data=pdf_data,
                           file_name=f"{sample.name}_masscalc.pdf", mime="application/pdf",
                           key=f"masscalc_pdf_{export_key}")
    plt.close(fig_calc)

    # Clean version: only calc & obs, no per-peak labels
    st.divider()
    fig_style_clean = {
        'fig_width': settings['fig_width'],
        'show_grid': False,
        'deconv_x_min_da': settings['deconv_x_min_da'],
        'deconv_x_max_da': settings['deconv_x_max_da'],
        'deconv_show_title': settings['deconv_show_title'],
        'deconv_show_subtitle': settings['deconv_show_subtitle'],
        'deconv_show_obs_calc': True,
        'deconv_calc_mass_da': theoretical_masses if len(theoretical_masses) > 1 else theoretical_mass,
        'deconv_show_peak_labels': False,
    }
    fig_clean = create_deconvoluted_masses_figure(sample.name, display_results, fig_style_clean)
    st.pyplot(fig_clean, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        png_data = export_figure(fig_clean, dpi=settings['export_dpi'])
        st.download_button("Download PNG", data=png_data,
                           file_name=f"{sample.name}_masscalc_clean.png", mime="image/png",
                           key=f"masscalc_clean_png_{export_key}")
    with col2:
        svg_data = export_figure_svg(fig_clean)
        st.download_button("Download SVG", data=svg_data,
                           file_name=f"{sample.name}_masscalc_clean.svg", mime="image/svg+xml",
                           key=f"masscalc_clean_svg_{export_key}")
    with col3:
        pdf_data = export_figure_pdf(fig_clean, dpi=settings['export_dpi'])
        st.download_button("Download PDF", data=pdf_data,
                           file_name=f"{sample.name}_masscalc_clean.pdf", mime="application/pdf",
                           key=f"masscalc_clean_pdf_{export_key}")
    plt.close(fig_clean)


def report_export_tab(sample_list: list, settings):
    """Generate a multi-page PDF report for a selected sample."""
    from matplotlib.backends.backend_pdf import PdfPages
    import io
    import datetime

    st.header("Report Export")

    # Sample selector
    if len(sample_list) == 1:
        sample = sample_list[0]
    else:
        selected_idx = st.selectbox(
            "Select sample",
            range(len(sample_list)),
            format_func=lambda i: sample_list[i].name,
            key="report_sample_idx"
        )
        sample = sample_list[selected_idx]

    if sample.error:
        st.error(f"Error loading sample: {sample.error}")
        return

    st.write(f"**Sample:** {sample.name}")
    st.write(f"**Method:** {sample.acq_method or 'N/A'}")

    # Options
    include_uv = st.checkbox("Include UV chromatogram", value=sample.uv_data is not None,
                             disabled=sample.uv_data is None, key="report_inc_uv")
    include_deconv = st.checkbox("Include deconvolution results", value=True, key="report_inc_deconv")

    # Check if deconv results are available for this sample
    deconv_results = None
    deconv_time_range = None
    if include_deconv:
        stored_results = st.session_state.get('deconv_results')
        stored_sample = st.session_state.get('deconv_current_sample')
        if stored_results and stored_sample == sample.name:
            deconv_results = stored_results
            deconv_time_range = st.session_state.get('deconv_time_range')
        else:
            # Try running auto deconvolution
            if sample.ms_scans is not None and sample.tic is not None:
                auto_start, auto_end = _detect_deconvolution_window(sample)
                if auto_end > auto_start:
                    mz, intensity = sum_spectra_in_range(sample, auto_start, auto_end)
                    if mz is not None and len(mz) > 0:
                        deconv_results = _run_default_deconvolution(mz, intensity)
                        deconv_time_range = (auto_start, auto_end)
            if not deconv_results:
                st.caption("No deconvolution results available. Run deconvolution first, or auto-deconv will be attempted.")

    A4_W, A4_H = 8.27, 11.69  # A4 in inches

    if st.button("Generate PDF Report", type="primary", key="report_generate"):
        with st.spinner("Generating report..."):
            buf = io.BytesIO()
            with PdfPages(buf) as pdf:
                # Page 1: Info + results table (already A4)
                params = {
                    'Mass range': '500 – 50,000 Da',
                    'Charge range': '1 – 50',
                    'Noise cutoff': '1,000 counts',
                }
                fig_info = create_report_info_page(
                    sample_name=sample.name,
                    acq_method=sample.acq_method,
                    app_version=config.APP_VERSION,
                    time_range=deconv_time_range,
                    parameters=params if deconv_results else {},
                    results=deconv_results,
                    acq_info=getattr(sample, 'acq_info', None),
                )
                pdf.savefig(fig_info)
                plt.close(fig_info)

                # Page 2: Chromatograms (UV + TIC)
                if sample.tic is not None or (include_uv and sample.uv_data is not None):
                    style = {
                        'fig_width': A4_W - 0.8,
                        'fig_height_per_panel': 3.0,
                        'line_width': settings['line_width'],
                        'show_grid': settings['show_grid'],
                        'y_scale': 'linear',
                        'colors': settings['colors'],
                        'labels': settings['labels'],
                    }
                    fig_chrom = create_single_sample_figure(
                        sample,
                        uv_wavelengths=settings['uv_wavelengths'] if include_uv else [],
                        eic_targets=[],
                        style=style,
                        uv_smoothing=settings['uv_smoothing'],
                        eic_smoothing=settings['eic_smoothing'],
                    )
                    fig_chrom.set_size_inches(A4_W, A4_H)
                    if fig_chrom._suptitle:
                        fig_chrom._suptitle.set_y(0.98)
                    fig_chrom.subplots_adjust(top=0.93)
                    pdf.savefig(fig_chrom)
                    plt.close(fig_chrom)

                # Page 3: Deconvolution figure (if results available)
                if deconv_results and deconv_time_range:
                    deconv_style = {
                        'fig_width': A4_W - 0.8,
                        'line_width': settings['line_width'],
                        'show_grid': True,
                        'deconv_x_min_da': settings['deconv_x_min_da'],
                        'deconv_x_max_da': settings['deconv_x_max_da'],
                        'deconv_show_obs_calc': False,
                        'deconv_calc_mass_da': None,
                    }
                    display_results = deconv_results[:10]
                    fig_deconv = create_deconvolution_figure(
                        sample, deconv_time_range[0], deconv_time_range[1],
                        display_results, deconv_style
                    )
                    fig_deconv.set_size_inches(A4_W, A4_H)
                    if fig_deconv._suptitle:
                        fig_deconv._suptitle.set_y(0.98)
                    fig_deconv.subplots_adjust(top=0.93)
                    pdf.savefig(fig_deconv)
                    plt.close(fig_deconv)

                    # Page 4: Ion selection per component
                    report_mz, report_intensity = sum_spectra_in_range(
                        sample, deconv_time_range[0], deconv_time_range[1]
                    )
                    if report_mz is not None and len(report_mz) > 0:
                        ion_style = {
                            'fig_width': A4_W - 0.8,
                            'line_width': settings['line_width'],
                            'show_grid': True,
                        }
                        fig_ions = create_ion_selection_figure(
                            report_mz, report_intensity,
                            display_results, ion_style
                        )
                        fig_ions.set_size_inches(A4_W, A4_H)
                        if fig_ions._suptitle:
                            fig_ions._suptitle.set_y(0.98)
                        fig_ions.subplots_adjust(top=0.93)
                        pdf.savefig(fig_ions)
                        plt.close(fig_ions)

            buf.seek(0)
            date_str = datetime.date.today().strftime("%Y%m%d")
            filename = f"{sample.name}_report_{date_str}.pdf"
            st.download_button(
                "Download PDF Report",
                data=buf.getvalue(),
                file_name=filename,
                mime="application/pdf",
                key="report_download"
            )
        st.success("Report generated successfully.")


def main():
    """Main application entry point."""
    init_session_state()

    st.title("LC-MS Analysis")
    st.caption(f"Version {config.APP_VERSION}")

    # Check rainbow availability
    if not check_rainbow_available():
        st.error("rainbow-api is not installed. Please install it with: pip install rainbow-api")
        return

    # Data source selector in sidebar
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Upload Files", "Browse Local"],
        index=0 if st.session_state.data_source == 'upload' else 1,
        horizontal=True
    )
    st.session_state.data_source = 'upload' if data_source == "Upload Files" else 'browse'

    # Sidebar - file selection based on mode
    if st.session_state.data_source == 'upload':
        selected_files = sidebar_file_upload()
    else:
        selected_files = sidebar_file_browser()

    # Load samples BEFORE settings so wavelengths can be detected
    if selected_files:
        load_samples(selected_files)

    settings = sidebar_settings()

    # Main content
    if not selected_files:
        if st.session_state.data_source == 'upload':
            st.info("Upload .D folder(s) as ZIP files using the sidebar to begin analysis.")
            st.markdown("""
            ### How to Upload
            1. **ZIP your .D folder** - Right-click on your `.D` folder and compress/zip it
            2. **Upload** - Use the file uploader in the sidebar
            3. **Analyze** - Select samples and view results

            ### Features
            - **Single Sample Analysis**: View UV, TIC, and EIC chromatograms
            - **Time Progression**: Compare 2+ samples across timepoints
            - **Time Change MS**: Compare summed mass spectra across 2+ samples
            - **EIC Batch Extraction**: Extract multiple EICs with peak area calculation
            - **Protein Deconvolution**: Deconvolute multiply-charged protein spectra
            - **Batch Deconvolution**: Run default deconvolution for multiple samples at once
            - **Mass Calculator**: Compare theoretical vs observed masses, annotate modifications
            - **Report Export**: Generate multi-page PDF reports
            - **Export**: Download plots as PNG, SVG, or PDF
            """)
        else:
            st.info("Select one or more .D folders from the sidebar to begin analysis.")
            st.markdown("""
            ### Features
            - **Single Sample Analysis**: View UV, TIC, and EIC chromatograms
            - **Time Progression**: Compare 2+ samples across timepoints
            - **Time Change MS**: Compare summed mass spectra across 2+ samples
            - **EIC Batch Extraction**: Extract multiple EICs with peak area calculation
            - **Protein Deconvolution**: Deconvolute multiply-charged protein spectra
            - **Batch Deconvolution**: Run default deconvolution for multiple samples at once
            - **Mass Calculator**: Compare theoretical vs observed masses, annotate modifications
            - **Report Export**: Generate multi-page PDF reports
            - **Export**: Download plots as PNG, SVG, or PDF
            """)
        return

    # Load samples
    samples = load_samples(selected_files)
    sample_list = [samples[p] for p in selected_files]

    # Tab selector using a single session-state key to avoid one-click lag.
    tab_options = ["Single Sample", "EIC Batch", "Deconvolution", "Batch Deconvolution", "Time Progression", "Time Change MS", "Mass Calculator", "Report Export"]

    # Migrate legacy tab key from older builds if present.
    if "active_tab" not in st.session_state:
        legacy_tab = st.session_state.get("tab_selector")
        st.session_state.active_tab = legacy_tab if legacy_tab in tab_options else "Single Sample"
    if "tab_selector" in st.session_state:
        del st.session_state["tab_selector"]
    if st.session_state.get("active_tab") == "Report":
        st.session_state.active_tab = "Report Export"

    # Auto-switch to Deconvolution tab for C4 method samples.
    # Do not override an explicit user tab choice (e.g., Batch/Time tabs).
    if len(sample_list) == 1 and getattr(sample_list[0], 'is_c4_method', False):
        current_sample_name = sample_list[0].name
        if st.session_state.get('_c4_auto_switched') != current_sample_name:
            if st.session_state.get("active_tab") in ("Single Sample", "Deconvolution"):
                st.session_state.active_tab = "Deconvolution"
            st.session_state._c4_auto_switched = current_sample_name

    if st.session_state.get("active_tab") not in tab_options:
        st.session_state.active_tab = "Single Sample"
    active_tab = st.radio(
        "Analysis",
        tab_options,
        horizontal=True,
        key="active_tab",
        label_visibility="collapsed"
    )

    st.divider()

    if active_tab == "Single Sample":
        if len(selected_files) == 1:
            single_sample_analysis(sample_list[0], settings)
        else:
            if st.session_state.get("single_sample_path") not in selected_files:
                st.session_state.single_sample_path = selected_files[0]
            selected_path = st.selectbox(
                "Select sample",
                selected_files,
                index=selected_files.index(st.session_state.single_sample_path),
                format_func=lambda p: samples[p].name,
                key="single_sample_select_path"
            )
            st.session_state.single_sample_path = selected_path
            single_sample_analysis(samples[selected_path], settings)

    elif active_tab == "EIC Batch":
        if len(selected_files) == 1:
            eic_batch_analysis(sample_list[0], settings)
        else:
            if st.session_state.get("eic_sample_path") not in selected_files:
                st.session_state.eic_sample_path = selected_files[0]
            selected_path = st.selectbox(
                "Select sample for EIC",
                selected_files,
                index=selected_files.index(st.session_state.eic_sample_path),
                format_func=lambda p: samples[p].name,
                key="eic_sample_select_path"
            )
            st.session_state.eic_sample_path = selected_path
            eic_batch_analysis(samples[selected_path], settings)

    elif active_tab == "Deconvolution":
        if len(selected_files) == 1:
            deconvolution_analysis(sample_list[0], settings)
        else:
            if st.session_state.get("deconv_sample_path") not in selected_files:
                st.session_state.deconv_sample_path = selected_files[0]
            selected_path = st.selectbox(
                "Select sample for Deconvolution",
                selected_files,
                index=selected_files.index(st.session_state.deconv_sample_path),
                format_func=lambda p: samples[p].name,
                key="deconv_sample_select_path"
            )
            st.session_state.deconv_sample_path = selected_path
            deconvolution_analysis(samples[selected_path], settings)

    elif active_tab == "Batch Deconvolution":
        if len(selected_files) == 1:
            st.info("Select multiple samples to enable Batch Deconvolution.")
        else:
            batch_deconvolution_analysis(sample_list, settings)

    elif active_tab == "Time Progression":
        if len(selected_files) == 1:
            st.info("Select multiple samples to enable Time Progression analysis.")
        else:
            time_progression_analysis(sample_list, settings)

    elif active_tab == "Time Change MS":
        if len(selected_files) == 1:
            st.info("Select multiple samples to enable Time Change MS.")
        else:
            time_change_mass_spectra_analysis(sample_list, settings)

    elif active_tab == "Mass Calculator":
        mass_calculator_tab(sample_list, settings)

    elif active_tab == "Report Export":
        report_export_tab(sample_list, settings)


if __name__ == "__main__":
    main()
