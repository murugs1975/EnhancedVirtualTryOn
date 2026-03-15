import React, { useState, useEffect, useCallback, useRef } from 'react';
import Head from 'next/head';
import toast from 'react-hot-toast';
import { RefreshCw, Wifi, Eye, Loader, Download, Maximize2, X, Info, ChevronDown, Power, Clock } from 'lucide-react';
import ImageUploader from '@/components/ImageUploader';
import LogPanel from '@/components/LogPanel';
import {
  performTryOn,
  getApiErrorMessage,
  TryOnInputs,
  fetchPipelinePreview,
  PipelinePreview,
} from '@/utils/api';

interface InputSlot {
  key: keyof TryOnInputs;
  label: string;
  description: string;
}

const INPUT_SLOTS: InputSlot[] = [
  { key: 'person', label: 'Person / Model', description: 'Full-body person image' },
  { key: 'cloth', label: 'Cloth', description: 'Target garment image to try on' },
];

export default function Home() {
  const [images, setImages] = useState<Record<string, File | null>>({
    person: null,
    cloth: null,
  });
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [resultUrl, setResultUrl] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState('');
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [modelsLoaded, setModelsLoaded] = useState<boolean | null>(null);
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [pipelinePreview, setPipelinePreview] = useState<PipelinePreview | null>(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [zoomImage, setZoomImage] = useState<{ url: string; title: string } | null>(null);
  const previewRequestRef = useRef(0);
  const autoGenerateRef = useRef(false);
  const [statusPanelOpen, setStatusPanelOpen] = useState(false);
  const [isWaking, setIsWaking] = useState(false);
  const healthIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const SESSION_TIMEOUT = 15 * 60 * 1000; // 15 minutes
  const [lastActivityTime, setLastActivityTime] = useState<number | null>(null);
  const [sessionRemaining, setSessionRemaining] = useState(0);
  const sessionExpired = useRef(false);

  // Reset session activity timer (called on user actions)
  const resetActivity = useCallback(() => {
    setLastActivityTime(Date.now());
    sessionExpired.current = false;
  }, []);

  // Health check with adaptive polling
  const checkHealth = useCallback(async () => {
    // Don't poll if session expired (let Cloud Run sleep)
    if (sessionExpired.current) return;
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8020';
      const res = await fetch(`${apiUrl}/health`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const data = await res.json();
        setBackendOnline(true);
        setModelsLoaded(data.model_loaded);
        if (data.model_loaded) {
          setIsWaking(false);
        }
      } else {
        setBackendOnline(false);
        setModelsLoaded(false);
      }
    } catch {
      setBackendOnline(false);
      setModelsLoaded(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = isWaking ? 3000 : 30000;
    if (healthIntervalRef.current) clearInterval(healthIntervalRef.current);
    healthIntervalRef.current = setInterval(checkHealth, interval);
    return () => {
      if (healthIntervalRef.current) clearInterval(healthIntervalRef.current);
    };
  }, [isWaking, checkHealth]);

  // Start session timer when backend comes online with models loaded
  useEffect(() => {
    if (backendOnline && modelsLoaded && !lastActivityTime) {
      resetActivity();
    }
    if (!backendOnline) {
      setLastActivityTime(null);
      setSessionRemaining(0);
    }
  }, [backendOnline, modelsLoaded]);

  // Countdown tick — updates every second
  useEffect(() => {
    if (!lastActivityTime || !backendOnline) return;
    const tick = () => {
      const elapsed = Date.now() - lastActivityTime;
      const remaining = Math.max(0, SESSION_TIMEOUT - elapsed);
      setSessionRemaining(remaining);
      if (remaining <= 0) {
        sessionExpired.current = true;
        // Stop health polling so Cloud Run can sleep
        if (healthIntervalRef.current) clearInterval(healthIntervalRef.current);
      }
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [lastActivityTime, backendOnline]);

  const handleWakeUp = useCallback(() => {
    setIsWaking(true);
    sessionExpired.current = false;
    resetActivity();
    checkHealth();
    toast('Waking up the server...', { icon: '\u{1F504}' });
  }, [checkHealth, resetActivity]);

  const pendingAutoPreview = useRef(false);
  const imagesRef = useRef(images);
  imagesRef.current = images;

  const handleImageSelect = (key: string, file: File) => {
    resetActivity();
    setImages(prev => ({ ...prev, [key]: file }));
    setPipelinePreview(null);
    setResultUrl('');
    pendingAutoPreview.current = true;
    // Invalidate any in-flight preview request so stale results are ignored
    previewRequestRef.current++;
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviews(prev => ({ ...prev, [key]: reader.result as string }));
    };
    reader.readAsDataURL(file);
  };

  const handleClearImage = (key: string) => {
    setImages(prev => ({ ...prev, [key]: null }));
    setPipelinePreview(null);
    setResultUrl('');
    // Invalidate any in-flight preview request
    previewRequestRef.current++;
    setIsPreviewLoading(false);
    setPreviews(prev => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  const allImagesUploaded = INPUT_SLOTS.every(slot => images[slot.key] !== null);

  // Auto-regenerate pipeline when images change and both are uploaded
  useEffect(() => {
    if (pendingAutoPreview.current && allImagesUploaded && !isProcessing) {
      pendingAutoPreview.current = false;
      // If a preview is already loading, it will be invalidated by the incremented requestRef
      handlePreview();
    }
  }, [images, allImagesUploaded, isPreviewLoading]);

  // Auto-trigger try-on generation after pipeline preview completes
  useEffect(() => {
    if (pipelinePreview && !resultUrl && !isProcessing && autoGenerateRef.current) {
      autoGenerateRef.current = false;
      handleGenerate();
    }
  }, [pipelinePreview]);

  const handlePreview = async () => {
    resetActivity();
    // Use ref to get latest images (avoids stale closure when called from useEffect)
    const currentImages = imagesRef.current;
    if (!currentImages.person || !currentImages.cloth) return;
    const requestId = ++previewRequestRef.current;
    setIsPreviewLoading(true);
    setPipelinePreview(null);
    setResultUrl('');
    autoGenerateRef.current = true;

    try {
      const preview = await fetchPipelinePreview(currentImages.person as File, currentImages.cloth as File);
      if (previewRequestRef.current === requestId) {
        setPipelinePreview(preview);
      }
    } catch (error) {
      if (previewRequestRef.current === requestId) {
        setPipelinePreview(null);
        autoGenerateRef.current = false;
        toast.error(`Pipeline preview failed: ${getApiErrorMessage(error)}`);
      }
    } finally {
      if (previewRequestRef.current === requestId) {
        setIsPreviewLoading(false);
      }
    }
  };

  const handleGenerate = async () => {
    if (!allImagesUploaded || isProcessing) return;
    resetActivity();
    setIsProcessing(true);
    setProcessingStage('Running cloth warping...');

    try {
      const inputs: TryOnInputs = {
        person: images.person!,
        cloth: images.cloth!,
      };
      setProcessingStage('Synthesizing final try-on...');
      const result = await performTryOn(inputs);
      setResultUrl(result);
      setProcessingStage('');
      toast.success('Try-on generated successfully!');
    } catch (error) {
      toast.error(getApiErrorMessage(error));
      setProcessingStage('');
    } finally {
      setIsProcessing(false);
    }
  };

  // Click-outside handler for status panel
  useEffect(() => {
    if (!statusPanelOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('[data-status-panel]')) {
        setStatusPanelOpen(false);
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [statusPanelOpen]);

  const confirmReset = () => {
    setImages({ person: null, cloth: null });
    setPreviews({});
    setPipelinePreview(null);
    setResultUrl('');
    setShowResetConfirm(false);
    autoGenerateRef.current = false;
    toast.success('Reset complete');
  };

  const uploadedCount = INPUT_SLOTS.filter(s => images[s.key] !== null).length;

  const canPreview = allImagesUploaded && !isPreviewLoading && !isProcessing
    && backendOnline !== false && modelsLoaded !== false;

  const handleDownload = () => {
    if (!resultUrl) return;
    const a = document.createElement('a');
    a.href = resultUrl;
    a.download = 'tryon_result.png';
    a.click();
  };

  const [activeTooltip, setActiveTooltip] = useState<string | null>(null);

  const pipelinePanels = [
    { title: 'Person (GT)', url: previews.person, loading: false,
      info: 'Your uploaded person photo, resized to 768\u00d71024. This is the ground-truth reference used throughout the pipeline.' },
    { title: 'Garment', url: previews.cloth, loading: false,
      info: 'The target clothing item you want to try on, resized to match the pipeline resolution.' },
    { title: 'Cloth Mask', url: pipelinePreview?.cloth_mask, loading: isPreviewLoading,
      info: 'Extracted garment silhouette (white = garment, black = background). Uses alpha channel for transparent images or GrabCut segmentation for opaque ones.' },
    { title: 'Agnostic', url: pipelinePreview?.agnostic, loading: isPreviewLoading,
      info: 'Person image with the clothing area masked out in gray. Face, hair, and lower body are preserved so the model knows where to place the new garment.' },
    { title: 'Warped Cloth', url: pipelinePreview?.warped_cloth, loading: isPreviewLoading,
      info: 'Garment deformed to match the person\u2019s body pose using optical flow predicted by the ConditionGenerator (Stage 1 ONNX model). Some distortion is normal \u2014 the final model refines it.' },
    { title: 'Try-On Result', url: resultUrl || undefined, loading: isProcessing,
      info: 'Final output from the SPADE generator neural network (Stage 2), which blends the warped garment seamlessly onto the person, fixing artifacts and adding realistic shading.' },
  ];

  return (
    <>
      <Head>
        <title>HR-VITON Virtual Try-On</title>
        <meta name="description" content="HR-VITON high-resolution virtual try-on" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-100">
        {/* Compact Header */}
        <header className="bg-gradient-to-r from-brand-600 to-brand-700 text-white shadow-xl sticky top-0 z-10">
          <div className="max-w-[1600px] mx-auto px-4 py-3 flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold">HR-VITON</h1>
              <p className="text-brand-100 text-xs">High-Resolution Virtual Try-On (ONNX)</p>
            </div>
            <div className="flex items-center gap-3">
              {/* Offline: prominent Start button directly in header */}
              {(backendOnline === false || backendOnline === null) && !isWaking && (
                <button
                  onClick={handleWakeUp}
                  className="flex items-center gap-2 bg-white/20 hover:bg-white/30 text-white px-4 py-1.5 rounded-lg font-medium text-sm transition-colors"
                >
                  <Power size={14} /> Start Virtual Try-On
                </button>
              )}
              {/* Waking: spinner in header */}
              {isWaking && !backendOnline && (
                <span className="flex items-center gap-1.5 text-xs font-medium text-yellow-200 bg-yellow-900/40 px-2.5 py-1 rounded-full">
                  <Loader size={12} className="animate-spin" /> Waking Up...
                </span>
              )}
              {/* Online: status badge with countdown + dropdown */}
              {backendOnline === true && (
                <div className="relative" data-status-panel>
                  <button
                    onClick={() => setStatusPanelOpen(!statusPanelOpen)}
                    className="flex items-center gap-1.5 text-sm"
                  >
                    {modelsLoaded === true ? (
                      <span className="flex items-center gap-1.5 text-xs font-medium text-green-200 bg-green-900/40 px-2.5 py-1 rounded-full">
                        <Wifi size={12} /> Ready
                        {sessionRemaining > 0 && (
                          <span className="text-green-300/80 ml-0.5">
                            <Clock size={10} className="inline mr-0.5" />
                            {Math.floor(sessionRemaining / 60000)}:{String(Math.floor((sessionRemaining % 60000) / 1000)).padStart(2, '0')}
                          </span>
                        )}
                        <ChevronDown size={10} className={`transition-transform duration-200 ${statusPanelOpen ? 'rotate-180' : ''}`} />
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5 text-xs font-medium text-yellow-200 bg-yellow-900/40 px-2.5 py-1 rounded-full">
                        <Loader size={12} className="animate-spin" /> Getting Ready
                        <ChevronDown size={10} className={`transition-transform duration-200 ${statusPanelOpen ? 'rotate-180' : ''}`} />
                      </span>
                    )}
                  </button>

                  {statusPanelOpen && (
                    <div className="absolute right-0 top-full mt-2 w-72 bg-white rounded-xl shadow-2xl border border-gray-200 p-4 z-20">
                      <h3 className="text-sm font-semibold text-gray-800 mb-3">Server Status</h3>
                      <div className="space-y-2 text-xs text-gray-600 mb-3">
                        <div className="flex justify-between">
                          <span>Container</span>
                          <span className="text-green-600 font-medium">Online</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Models</span>
                          <span className={modelsLoaded ? 'text-green-600 font-medium' : 'text-gray-400 font-medium'}>
                            {modelsLoaded ? 'Loaded' : 'Loading...'}
                          </span>
                        </div>
                        {modelsLoaded && sessionRemaining > 0 && (
                          <div className="flex justify-between">
                            <span>Session timeout</span>
                            <span className={`font-medium ${sessionRemaining < 120000 ? 'text-red-500' : 'text-brand-600'}`}>
                              {Math.floor(sessionRemaining / 60000)}m {Math.floor((sessionRemaining % 60000) / 1000)}s
                            </span>
                          </div>
                        )}
                      </div>
                      {modelsLoaded && (
                        <p className="text-xs text-gray-400 text-center">
                          {sessionRemaining > 0
                            ? 'Timer resets on each action (upload, preview, try-on)'
                            : 'Session expired — server will sleep soon'}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
              <button
                onClick={() => {
                  if (uploadedCount > 0 || resultUrl) setShowResetConfirm(true);
                  else toast.success('Nothing to reset');
                }}
                className="flex items-center gap-1.5 bg-white/20 hover:bg-white/30 px-3 py-1.5 rounded-lg transition-colors text-sm"
              >
                <RefreshCw size={14} /> Reset
              </button>
            </div>
          </div>
        </header>

        {/* Offline banner */}
        {(backendOnline === false || backendOnline === null) && !isWaking && (
          <div className="bg-gray-50 border-b border-gray-200 py-8 text-center">
            <Power size={40} className="mx-auto text-brand-500 mb-3" />
            <h2 className="text-lg font-semibold text-gray-800 mb-2">Server is sleeping</h2>
            <p className="text-sm text-gray-500 mb-4">Click below to wake up the server (takes 30-90 seconds)</p>
            <button
              onClick={handleWakeUp}
              className="btn-primary px-6 py-3 text-base inline-flex items-center gap-2"
            >
              <Power size={18} /> Start Virtual Try-On
            </button>
          </div>
        )}
        {isWaking && !backendOnline && (
          <div className="bg-yellow-50 border-b border-yellow-200 py-6 text-center">
            <Loader size={32} className="mx-auto text-yellow-600 mb-2 animate-spin" />
            <h2 className="text-lg font-semibold text-yellow-800">Waking up the server...</h2>
            <p className="text-sm text-yellow-600">This typically takes 30-90 seconds. Models are loading.</p>
          </div>
        )}

        <div className="max-w-[1600px] mx-auto px-4 py-4">
          {/* Upload row + Generate button - compact horizontal layout */}
          <section className="mb-4">
            <div className="flex flex-col lg:flex-row items-start lg:items-end gap-4">
              <div className="flex-1 grid grid-cols-1 sm:grid-cols-2 gap-4 w-full">
                {INPUT_SLOTS.map((slot) => (
                  <div key={slot.key} className="glass-card p-2">
                    <ImageUploader
                      label={slot.label}
                      description={slot.description}
                      guidance=""
                      onImageSelect={(file) => handleImageSelect(slot.key, file)}
                      previewUrl={previews[slot.key] || ''}
                      onClear={() => handleClearImage(slot.key)}
                      disabled={isProcessing || isPreviewLoading || !backendOnline || !modelsLoaded}
                      disabledMessage={!backendOnline || !modelsLoaded ? 'Start the server to upload images' : undefined}
                    />
                  </div>
                ))}
              </div>
              {allImagesUploaded && (
                <button
                  onClick={handlePreview}
                  disabled={!canPreview}
                  className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed px-6 py-3 text-base whitespace-nowrap flex items-center gap-2 shrink-0"
                >
                  {isPreviewLoading ? (
                    <><Loader size={18} className="animate-spin" /> Generating...</>
                  ) : (
                    <><Eye size={18} /> Generate Pipeline Preview</>
                  )}
                </button>
              )}
            </div>
          </section>

          {/* Pipeline preview - 6 images, full width */}
          <section className="mb-4">
            <h2 className="text-lg font-bold text-gray-800 mb-3 text-center">
              Data-Pipeline Preview
              {isProcessing && (
                <span className="ml-3 text-sm font-normal text-brand-600">
                  <Loader size={14} className="inline animate-spin mr-1" />
                  {processingStage || 'Generating try-on...'}
                </span>
              )}
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {pipelinePanels.map((panel) => (
                <div key={panel.title} className="glass-card p-2 relative group">
                  <div className="flex items-center justify-center gap-1 mb-1.5 relative">
                    <p className="text-xs font-semibold text-gray-600">{panel.title}</p>
                    <button
                      onClick={(e) => { e.stopPropagation(); setActiveTooltip(activeTooltip === panel.title ? null : panel.title); }}
                      onMouseEnter={() => setActiveTooltip(panel.title)}
                      onMouseLeave={() => setActiveTooltip(null)}
                      className="text-gray-400 hover:text-brand-600 transition-colors shrink-0"
                    >
                      <Info size={12} />
                    </button>
                    {activeTooltip === panel.title && (
                      <div className="absolute top-full left-1/2 -translate-x-1/2 mt-1 z-20 w-56 bg-gray-800 text-white text-[11px] leading-relaxed rounded-lg shadow-lg px-3 py-2 pointer-events-none">
                        <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-gray-800 rotate-45" />
                        {panel.info}
                      </div>
                    )}
                  </div>
                  <div className="w-full aspect-[3/4] rounded-lg border border-gray-200 bg-white overflow-hidden flex items-center justify-center relative">
                    {panel.url ? (
                      <>
                        <img
                          src={panel.url}
                          alt={panel.title}
                          className="w-full h-full object-contain"
                        />
                        {/* Zoom button on hover */}
                        <button
                          onClick={() => setZoomImage({ url: panel.url!, title: panel.title })}
                          className="absolute top-1 right-1 bg-black/50 hover:bg-black/70 text-white p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <Maximize2 size={14} />
                        </button>
                        {/* Download on try-on result */}
                        {panel.title === 'Try-On Result' && (
                          <button
                            onClick={handleDownload}
                            className="absolute bottom-1 right-1 bg-brand-600 hover:bg-brand-700 text-white px-2 py-1 rounded text-xs flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
                          >
                            <Download size={12} /> Save
                          </button>
                        )}
                      </>
                    ) : panel.loading ? (
                      <div className="flex flex-col items-center gap-1.5">
                        <Loader size={18} className="animate-spin text-brand-500" />
                        <p className="text-xs text-gray-500">
                          {panel.title === 'Try-On Result' ? 'Synthesizing...' : 'Processing...'}
                        </p>
                      </div>
                    ) : (
                      <p className="text-xs text-gray-400 text-center px-2">
                        {panel.title === 'Person (GT)' || panel.title === 'Garment'
                          ? 'Upload image above'
                          : panel.title === 'Try-On Result'
                          ? pipelinePreview ? 'Auto-generating...' : 'Waiting for preview'
                          : 'Run pipeline preview'}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Log Panel - collapsed */}
            {pipelinePreview?.logs && pipelinePreview.logs.length > 0 && (
              <LogPanel logs={pipelinePreview.logs} />
            )}
          </section>

          {/* Footer */}
          <footer className="border-t border-white/20 pt-4 text-center text-gray-500 text-xs">
            <p>HR-VITON - High-Resolution Virtual Try-On</p>
          </footer>
        </div>

        {/* Zoom Modal */}
        {zoomImage && (
          <div
            className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
            onClick={() => setZoomImage(null)}
          >
            <div className="relative max-w-4xl max-h-[90vh] w-full" onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-white font-semibold">{zoomImage.title}</h3>
                <button
                  onClick={() => setZoomImage(null)}
                  className="text-white/70 hover:text-white p-1"
                >
                  <X size={20} />
                </button>
              </div>
              <img
                src={zoomImage.url}
                alt={zoomImage.title}
                className="w-full h-auto max-h-[80vh] object-contain rounded-lg bg-white"
              />
              {zoomImage.title === 'Try-On Result' && resultUrl && (
                <button
                  onClick={handleDownload}
                  className="mt-3 mx-auto flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 rounded-lg"
                >
                  <Download size={16} /> Download Result
                </button>
              )}
            </div>
          </div>
        )}

        {/* Reset Modal */}
        {showResetConfirm && (
          <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
            <div className="bg-white rounded-xl shadow-2xl p-6 max-w-sm w-full">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Reset Everything?</h3>
              <p className="text-sm text-gray-600 mb-6">
                This will clear all uploaded images and results.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => setShowResetConfirm(false)}
                  className="flex-1 px-4 py-2 rounded-lg bg-gray-100 text-gray-700 hover:bg-gray-200 font-medium transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmReset}
                  className="flex-1 px-4 py-2 rounded-lg bg-red-500 text-white hover:bg-red-600 font-medium transition-colors"
                >
                  Reset
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </>
  );
}
