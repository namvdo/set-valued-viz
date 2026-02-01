/**
 * Video Encoder Web Worker
 * Uses WebCodecs API for H264 encoding with mp4-muxer for container
 */

// Import mp4-muxer from CDN (worker-safe)
importScripts('https://cdn.jsdelivr.net/npm/mp4-muxer@5.2.2/build/mp4-muxer.min.js');

let muxer = null;
let videoEncoder = null;
let frameCount = 0;
let encodedChunks = [];

self.onmessage = async function (e) {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            await initEncoder(data);
            break;
        case 'frame':
            await encodeFrame(data);
            break;
        case 'finish':
            await finishEncoding();
            break;
        case 'abort':
            abortEncoding();
            break;
    }
};

async function initEncoder(config) {
    const { width, height, fps, filename } = config;

    frameCount = 0;
    encodedChunks = [];

    try {
        // Check WebCodecs support
        if (typeof VideoEncoder === 'undefined') {
            throw new Error('WebCodecs API not supported in this browser');
        }

        // Initialize mp4-muxer
        muxer = new Mp4Muxer.Muxer({
            target: new Mp4Muxer.ArrayBufferTarget(),
            video: {
                codec: 'avc',
                width: width,
                height: height,
            },
            fastStart: 'in-memory',
        });

        videoEncoder = new VideoEncoder({
            output: (chunk, meta) => {
                muxer.addVideoChunk(chunk, meta);
            },
            error: (err) => {
                console.error('VideoEncoder error:', err);
                self.postMessage({ type: 'error', error: err.message });
            }
        });

        // Configure encoder - using AVC (H.264) baseline profile
        await videoEncoder.configure({
            codec: 'avc1.42001f', // H.264 baseline profile level 3.1
            width: width,
            height: height,
            bitrate: 5_000_000, // 5 Mbps for good quality
            framerate: fps,
            latencyMode: 'quality',
            avc: { format: 'avc' },
        });

        self.postMessage({ type: 'ready' });
    } catch (err) {
        console.error('Failed to initialize encoder:', err);
        self.postMessage({ type: 'error', error: err.message });
    }
}

async function encodeFrame(data) {
    const { imageData, timestamp, duration } = data;

    if (!videoEncoder || videoEncoder.state !== 'configured') {
        self.postMessage({ type: 'error', error: 'Encoder not ready' });
        return;
    }

    try {
        // Create VideoFrame from ImageBitmap
        const frame = new VideoFrame(imageData, {
            timestamp: timestamp,
            duration: duration,
        });

        // Encode frame (keyframe on every frame for low FPS videos)
        const keyFrame = true;
        videoEncoder.encode(frame, { keyFrame });
        frame.close();

        frameCount++;
        self.postMessage({ type: 'progress', frameCount });
    } catch (err) {
        console.error('Frame encoding error:', err);
        self.postMessage({ type: 'error', error: err.message });
    }
}

async function finishEncoding() {
    if (!videoEncoder || !muxer) {
        self.postMessage({ type: 'error', error: 'Encoder not initialized' });
        return;
    }

    try {
        await videoEncoder.flush();
        videoEncoder.close();

        muxer.finalize();

        const buffer = muxer.target.buffer;
        const blob = new Blob([buffer], { type: 'video/mp4' });

        self.postMessage({
            type: 'complete',
            blob: blob,
            frameCount: frameCount
        });
    } catch (err) {
        console.error('Finalization error:', err);
        self.postMessage({ type: 'error', error: err.message });
    }
}

function abortEncoding() {
    try {
        if (videoEncoder && videoEncoder.state !== 'closed') {
            videoEncoder.close();
        }
        muxer = null;
        videoEncoder = null;
        frameCount = 0;
    } catch (err) {
        console.error('Abort error:', err);
    }
    self.postMessage({ type: 'aborted' });
}
