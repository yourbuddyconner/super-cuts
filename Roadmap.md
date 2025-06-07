# Super Cuts Roadmap

This document outlines the planned features and future direction for Super Cuts. The project is envisioned to evolve through several major versions, each building upon the last to serve a wider range of users, from individual creators to professional teams.

---

## Version 1.x: The Ultimate CLI Tool

The focus of v1.x is to make the open-source CLI an indispensable and powerful tool for individual video editors and power users. This version is all about stability, workflow integration, and providing more control to the user.

-   [x] **Editing Workflow Integration (XML/EDL Export)**
    -   Generate an XML or EDL file that can be imported directly into Adobe Premiere Pro, Final Cut Pro, or DaVinci Resolve. This file will create a timeline with markers or a sequence of the generated clips, fitting seamlessly into professional post-production workflows.
    -   **Implementation Plan:**
        -   **Format Choice:** We will generate a Final Cut Pro XML (`.fcpxml`) file. This format is feature-rich and widely supported by major video editors, including Premiere Pro and DaVinci Resolve.
        -   **Core Library:** We'll use the `OpenTimelineIO` (OTIO) Python library. It's an open-source industry standard for editorial timeline information, developed by Pixar.
        -   **Key Steps:**
            1.  **Add Dependency:** Integrate `opentimelineio` into the project.
            2.  **Create Exporter Module:** Develop a new function that takes the list of generated clips (moments) as input.
            3.  **Build Timeline:** Use OTIO to programmatically construct a timeline. This involves creating a `Timeline` object, adding a `Track`, and then populating the track with `Clip` objects that reference the source video file with the correct in/out points for each moment.
            4.  **Write FCPXML File:** Use the `otio.adapters.write_to_file()` method with the `fcpx_xml` adapter to serialize the timeline into an `.fcpxml` file.
            5.  **Add CLI Flag:** Introduce a new command-line argument, such as `--export-xml <filename>.fcpxml`, to enable this feature.

-   [x] **Local Transcription Support**
    -   Integrate with offline, local transcription models (e.g., `insanely-fast-whisper` via Hugging Face Transformers) to provide a free, private, and fast alternative to the OpenAI Whisper API.

-   [x] **Local Moment Analysis**
    -   Integrate with local, multi-modal LLMs (e.g., `Qwen2.5-VL`) to provide a free, private, and powerful alternative to the OpenAI GPT-4 Vision API for analyzing video and transcript data.
    -   Includes switchable analysis modes (`keyframes` vs. `video_clip`) for flexibility.

-   [ ] **Batch Processing**
    -   Process multiple video files at once with a single command (e.g., `supercuts video1.mp4 video2.mp4 video3.mp4`).
    -   Support folder input to process all video files in a directory (e.g., `supercuts /path/to/videos/`).
    -   Generate separate output folders for each video or combine results into a single project.
    -   Option to generate a master FCPXML file that includes clips from all processed videos.

-   [ ] **Scene Cut Detection**
    -   Use FFmpeg's scene detection capabilities (`scdet`) to automatically identify visual cuts in the video.
    -   Provide an alternative mode of operation (e.g., `supercuts video.mp4 --mode scene-detect`) to generate a clip for each detected shot.
    -   This offers a purely visual way to segment a video, complementing the AI-based transcript analysis.

-   [ ] **Granular Moment Control**
    -   Allow users to guide the AI with CLI flags to specify the types of moments they are interested in (e.g., `supercuts video.mp4 --find "vows, toasts"`).

-   [ ] **Configuration File Support**
    -   Allow users to define settings in a `supercuts.config.json` file (or similar) to avoid passing numerous flags for every run.

-   [ ] **Expanded Output Options**
    -   Add flags to export transcripts in different formats (`.srt`, `.vtt`, `.txt`).
    -   Add options to create simple audiogram clips for social media.

---

## Version 2.x: The "Pro" Experience

This version marks a significant leap, potentially introducing a premium desktop application or a web-based SaaS ("Super Cuts Pro"). The focus is on providing a rich user interface and more advanced AI-powered capabilities.

-   [ ] **Rich UI (Web or Desktop)**
    -   A visual interface to upload and manage videos.
    -   A video player synchronized with the transcript, allowing users to click on text to jump to that point in the video.

-   [ ] **Interactive Timeline Editor**
    -   A visual timeline where users can see the AI-suggested clips.
    -   Allow users to manually adjust the in/out points of clips, merge or split segments, and create new clips by highlighting text.

-   [ ] **Speaker Identification (Diarization)**
    -   Automatically detect and label different speakers in the video. Clips and transcript segments would be tagged by speaker (e.g., "Toast from Speaker 1").

-   [ ] **Visual Moment Detection (Experimental)**
    -   Explore computer vision models to identify non-verbal moments, such as detecting all shots of a specific object (e.g., a wedding cake) or identifying moments of high action or audience applause.

---

## Version 3.x: Collaboration & Cloud

Version 3.x expands Super Cuts into a tool for teams and organizations, focusing on cloud integration, collaboration, and turning video archives into searchable knowledge bases.

-   [ ] **Collaboration Features**
    -   Support for multi-user projects.
    -   Ability to share projects, leave comments on clips, and create a review/approval workflow.

-   [ ] **Cloud Storage Integration**
    -   Connect to Google Drive, Dropbox, or S3 to automatically process new videos added to a specific folder.

-   [ ] **API Access**
    -   Provide a developer API to allow programmatic submission of videos and retrieval of results, enabling integration into larger media asset management (MAM) systems.

-   [ ] **Advanced Search & Analytics**
    -   For organizations with large video libraries, provide a powerful search engine to find moments across all processed videos (e.g., "Find every time 'Q4 earnings' was mentioned"). 