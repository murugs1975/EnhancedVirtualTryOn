import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Download, Loader, ZoomIn, ZoomOut, X, Maximize2 } from 'lucide-react';
import clsx from 'clsx';

interface ResultViewerProps {
  originalUrl?: string;
  resultUrl?: string;
  isProcessing?: boolean;
  processingProgress?: { stage: string; percent: number };
}

type ViewMode = 'slider' | 'side-by-side' | 'result-only';

const ResultViewer: React.FC<ResultViewerProps> = ({
  originalUrl,
  resultUrl,
  isProcessing = false,
  processingProgress,
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('slider');
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [showZoomModal, setShowZoomModal] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const sliderRef = useRef<HTMLDivElement>(null);

  const updateSlider = useCallback((clientX: number) => {
    if (!sliderRef.current) return;
    const rect = sliderRef.current.getBoundingClientRect();
    const newPosition = ((clientX - rect.left) / rect.width) * 100;
    setSliderPosition(Math.max(1, Math.min(99, newPosition)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    updateSlider(e.clientX);
  }, [updateSlider]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isDragging) {
      updateSlider(e.clientX);
    }
  }, [isDragging, updateSlider]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    setIsDragging(true);
    updateSlider(e.touches[0].clientX);
  }, [updateSlider]);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (isDragging) {
      updateSlider(e.touches[0].clientX);
    }
  }, [isDragging, updateSlider]);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Zoom modal handlers
  const handleZoomWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoomLevel(prev => Math.max(0.5, Math.min(5, prev + (e.deltaY > 0 ? -0.2 : 0.2))));
  }, []);

  const handlePanStart = useCallback((e: React.MouseEvent) => {
    if (zoomLevel > 1) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
    }
  }, [zoomLevel, panOffset]);

  const handlePanMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) {
      setPanOffset({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    }
  }, [isPanning, panStart]);

  const handlePanEnd = useCallback(() => {
    setIsPanning(false);
  }, []);

  const openZoom = () => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
    setShowZoomModal(true);
  };

  const downloadResult = async () => {
    if (!resultUrl) return;
    try {
      const response = await fetch(resultUrl);
      const blob = await response.blob();
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `tryon-result-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  if (isProcessing) {
    const progress = processingProgress || { stage: 'Processing...', percent: 0 };
    return (
      <div className="w-full min-h-[300px] max-h-[600px] h-[50vh] glass-card flex flex-col items-center justify-center">
        <Loader size={48} className="text-brand-500 animate-spin mb-4" />
        <p className="text-lg font-semibold text-gray-800">
          {progress.stage}
        </p>
        {progress.percent > 0 && (
          <div className="w-64 mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-brand-500 h-2.5 rounded-full transition-all duration-500"
                style={{ width: `${progress.percent}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 mt-1 text-center">{Math.round(progress.percent)}%</p>
          </div>
        )}
        <p className="text-sm text-gray-600 mt-2">
          This may take a minute depending on image size
        </p>
      </div>
    );
  }

  if (!resultUrl) {
    return (
      <div className="w-full min-h-[300px] max-h-[600px] h-[50vh] glass-card flex flex-col items-center justify-center">
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-800 mb-2">
            No result yet
          </p>
          <p className="text-sm text-gray-600">
            Upload images and click &quot;Generate Try-On&quot; to see results
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="glass-card mb-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          View Mode
        </h3>
        <div className="flex gap-2 flex-wrap">
          {(['slider', 'side-by-side', 'result-only'] as ViewMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              className={clsx(
                'px-4 py-2 rounded-lg font-medium transition-all',
                viewMode === mode
                  ? 'bg-brand-500 text-white'
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              )}
            >
              {mode === 'slider' ? 'Slider' : mode === 'side-by-side' ? 'Side by Side' : 'Result Only'}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-card">
        {viewMode === 'slider' && originalUrl && (
          <div
            ref={sliderRef}
            className="relative w-full min-h-[300px] max-h-[600px] h-[50vh] overflow-hidden rounded-lg cursor-col-resize bg-gray-200 select-none"
            onMouseDown={handleMouseDown}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={() => setIsDragging(false)}
          >
            {/* Background: Original image */}
            <img
              src={originalUrl}
              alt="Original"
              className="absolute inset-0 w-full h-full object-contain"
              draggable={false}
            />
            {/* Foreground: Result image clipped to slider position */}
            <div
              className="absolute inset-y-0 left-0 overflow-hidden"
              style={{ width: `${sliderPosition}%` }}
            >
              <img
                src={resultUrl}
                alt="Result"
                className="absolute inset-0 h-full object-contain"
                style={{
                  width: sliderRef.current
                    ? `${sliderRef.current.getBoundingClientRect().width}px`
                    : '100vw',
                }}
                draggable={false}
              />
            </div>
            {/* Slider handle */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
              style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
            >
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white rounded-full p-2 shadow-lg">
                <div className="flex gap-0.5">
                  <div className="w-0.5 h-4 bg-brand-500"></div>
                  <div className="w-0.5 h-4 bg-brand-500"></div>
                </div>
              </div>
            </div>
            {/* Labels */}
            <div className="absolute bottom-4 left-4 bg-black/60 text-white px-3 py-1 rounded text-sm font-medium pointer-events-none">
              Original
            </div>
            <div className="absolute bottom-4 right-4 bg-black/60 text-white px-3 py-1 rounded text-sm font-medium pointer-events-none">
              Result
            </div>
            {/* Zoom button */}
            <button
              onClick={(e) => { e.stopPropagation(); openZoom(); }}
              className="absolute top-4 right-4 bg-black/50 hover:bg-black/70 text-white p-2 rounded-lg transition-colors"
              onMouseDown={(e) => e.stopPropagation()}
            >
              <Maximize2 size={18} />
            </button>
          </div>
        )}

        {viewMode === 'side-by-side' && originalUrl && (
          <div className="flex flex-col sm:flex-row gap-4 w-full">
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-600 mb-2">Original</p>
              <img
                src={originalUrl}
                alt="Original"
                className="w-full min-h-[200px] max-h-[500px] object-contain rounded-lg bg-gray-100"
              />
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-600 mb-2">Result</p>
              <div className="relative">
                <img
                  src={resultUrl}
                  alt="Result"
                  className="w-full min-h-[200px] max-h-[500px] object-contain rounded-lg bg-gray-100"
                />
                <button
                  onClick={openZoom}
                  className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 text-white p-2 rounded-lg transition-colors"
                >
                  <Maximize2 size={16} />
                </button>
              </div>
            </div>
          </div>
        )}

        {viewMode === 'result-only' && (
          <div>
            <p className="text-sm font-medium text-gray-600 mb-2">Result</p>
            <div className="relative">
              <img
                src={resultUrl}
                alt="Result"
                className="w-full min-h-[300px] max-h-[600px] object-contain rounded-lg bg-gray-100"
              />
              <button
                onClick={openZoom}
                className="absolute top-4 right-4 bg-black/50 hover:bg-black/70 text-white p-2 rounded-lg transition-colors"
              >
                <Maximize2 size={18} />
              </button>
            </div>
          </div>
        )}

        {resultUrl && (
          <button
            onClick={downloadResult}
            className="mt-4 w-full btn-primary flex items-center justify-center gap-2"
          >
            <Download size={20} />
            Download Result
          </button>
        )}
      </div>

      {/* Zoom Modal */}
      {showZoomModal && resultUrl && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center">
          <div className="absolute top-4 right-4 flex gap-2 z-10">
            <button
              onClick={() => setZoomLevel(prev => Math.max(0.5, prev - 0.25))}
              className="bg-white/20 hover:bg-white/30 text-white p-2 rounded-lg transition-colors"
            >
              <ZoomOut size={20} />
            </button>
            <span className="bg-white/20 text-white px-3 py-2 rounded-lg text-sm font-medium">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              onClick={() => setZoomLevel(prev => Math.min(5, prev + 0.25))}
              className="bg-white/20 hover:bg-white/30 text-white p-2 rounded-lg transition-colors"
            >
              <ZoomIn size={20} />
            </button>
            <button
              onClick={() => setShowZoomModal(false)}
              className="bg-white/20 hover:bg-white/30 text-white p-2 rounded-lg transition-colors"
            >
              <X size={20} />
            </button>
          </div>
          <div
            className="w-full h-full overflow-hidden cursor-grab active:cursor-grabbing"
            onWheel={handleZoomWheel}
            onMouseDown={handlePanStart}
            onMouseMove={handlePanMove}
            onMouseUp={handlePanEnd}
            onMouseLeave={handlePanEnd}
          >
            <img
              src={resultUrl}
              alt="Result zoomed"
              className="max-w-none select-none"
              style={{
                transform: `translate(calc(-50% + ${panOffset.x}px), calc(-50% + ${panOffset.y}px)) scale(${zoomLevel})`,
                position: 'absolute',
                top: '50%',
                left: '50%',
                maxHeight: '90vh',
                maxWidth: '90vw',
              }}
              draggable={false}
            />
          </div>
          <p className="absolute bottom-4 left-1/2 -translate-x-1/2 text-white/60 text-sm">
            Scroll to zoom. Drag to pan.
          </p>
        </div>
      )}
    </div>
  );
};

export default ResultViewer;
