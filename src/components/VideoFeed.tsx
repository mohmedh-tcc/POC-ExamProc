import React, { useRef, useEffect } from 'react';
import { NormalizedLandmark } from '../lib/math/iris-gaze';

interface VideoFeedProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  keyPoints: NormalizedLandmark[] | null;
  objectDetections: Array<{
    categoryName: string;
    score: number;
    boundingBox?: {
      originX: number;
      originY: number;
      width: number;
      height: number;
    };
  }>;
  showMesh: boolean;
  onToggleMesh: () => void;
  faceCount: number;
}

export const VideoFeed: React.FC<VideoFeedProps> = ({
  videoRef,
  keyPoints,
  objectDetections,
  showMesh,
  onToggleMesh,
  faceCount,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (showMesh && keyPoints && keyPoints.length > 0) {
      ctx.fillStyle = '#38bdf8';
      for (const pt of keyPoints) {
        ctx.beginPath();
        ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    // Draw object detection boxes
    for (const det of objectDetections) {
      if (!det.boundingBox) continue;
      const { originX, originY, width, height } = det.boundingBox;

      const isPhone = det.categoryName.toLowerCase() === 'cell phone';
      const isPerson = det.categoryName.toLowerCase() === 'person';

      ctx.lineWidth = 2;
      ctx.strokeStyle = isPhone ? '#ef4444' : isPerson ? '#f59e0b' : '#38bdf8';
      ctx.fillStyle = ctx.strokeStyle;

      ctx.strokeRect(originX, originY, width, height);
      ctx.font = '12px Inter, sans-serif';
      const label = `${det.categoryName} (${(det.score * 100).toFixed(0)}%)`;
      ctx.save();
      ctx.scale(-1, 1);
      ctx.fillText(label, -(originX + width - 4), Math.max(14, originY - 4));
      ctx.restore();
    }
  }, [keyPoints, objectDetections, showMesh, videoRef]);

  return (
    <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-slate-900 shadow-xl aspect-[4/3] max-w-[640px] w-full mx-auto">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover transform -scale-x-100"
      />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="absolute inset-0 w-full h-full object-cover transform -scale-x-100 pointer-events-none"
      />

      {/* Floating HUD Badges */}
      <div className="absolute top-3 left-3 flex items-center gap-2">
        <span className={`px-2.5 py-1 rounded-md text-xs font-semibold backdrop-blur-md ${
          faceCount === 1
            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
            : faceCount > 1
            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
            : 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
        }`}>
          {faceCount === 0 ? 'No Face' : faceCount === 1 ? '1 Face Detected' : `${faceCount} Faces Detected`}
        </span>
      </div>

      <div className="absolute bottom-3 right-3 flex items-center gap-2">
        <button
          onClick={onToggleMesh}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition backdrop-blur-md border ${
            showMesh
              ? 'bg-sky-500/20 border-sky-400 text-sky-300'
              : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200'
          }`}
        >
          {showMesh ? 'Hide Mesh' : 'Show Mesh'}
        </button>
      </div>
    </div>
  );
};
