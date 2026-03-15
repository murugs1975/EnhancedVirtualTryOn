import React, { useState } from 'react';
import { ChevronDown, ChevronRight, AlertTriangle, Info, AlertCircle, Cpu, Scissors, Move, Eye, Layers, ImageIcon } from 'lucide-react';

export interface PipelineLogEntry {
  step: string;
  level: 'info' | 'warn' | 'error';
  message: string;
}

interface LogPanelProps {
  logs: PipelineLogEntry[];
}

interface StageInfo {
  label: string;
  description: string;
  icon: React.ReactNode;
  color: string;
}

const STAGE_INFO: Record<string, StageInfo> = {
  preprocessing: {
    label: 'Input Preprocessing',
    description: 'Resize images, detect pose landmarks (MediaPipe PoseLandmarker heavy model, 33 keypoints), extract person segmentation mask.',
    icon: <ImageIcon size={14} />,
    color: 'text-blue-600 bg-blue-50 border-blue-200',
  },
  cloth_mask: {
    label: 'Cloth Mask Extraction',
    description: 'Extract garment foreground mask using alpha channel (if transparent) or GrabCut segmentation (for opaque images). Falls back to luminance thresholding.',
    icon: <Scissors size={14} />,
    color: 'text-purple-600 bg-purple-50 border-purple-200',
  },
  pose_detection: {
    label: 'Pose Detection',
    description: 'MediaPipe PoseLandmarker detects 33 body landmarks + outputs a segmentation mask. Landmarks drive all subsequent body-part region construction.',
    icon: <Eye size={14} />,
    color: 'text-cyan-600 bg-cyan-50 border-cyan-200',
  },
  densepose: {
    label: 'DensePose Proxy',
    description: 'Builds a colored body-part surface map (proxy for Detectron2 DensePose) using landmark positions. Each body part gets a specific color matching the training data palette.',
    icon: <Layers size={14} />,
    color: 'text-teal-600 bg-teal-50 border-teal-200',
  },
  parse_agnostic: {
    label: 'Parse-Agnostic Map',
    description: 'Constructs a 20-class CIHP label map from landmarks (hair, face, upper-clothes, pants, arms, etc.), then removes upper-body clothing labels (torso, arms, neck) to create a 13-channel one-hot agnostic parse map.',
    icon: <Layers size={14} />,
    color: 'text-indigo-600 bg-indigo-50 border-indigo-200',
  },
  agnostic: {
    label: 'Agnostic Person Image',
    description: 'Masks the person\'s torso, arms, and neck with gray (0.5) to remove clothing information. Uses morphological closing + dilation for gap-free coverage, then Gaussian feathering for smooth edges.',
    icon: <ImageIcon size={14} />,
    color: 'text-gray-600 bg-gray-50 border-gray-200',
  },
  downsampling: {
    label: 'Downsampling',
    description: 'All inputs are downsampled from fine resolution (768x1024) to low resolution (192x256) for the ConditionGenerator model. Bilinear for images, nearest-neighbor for masks/labels.',
    icon: <Move size={14} />,
    color: 'text-orange-600 bg-orange-50 border-orange-200',
  },
  tocg_model: {
    label: 'Stage 1: ConditionGenerator (tocg.onnx)',
    description: 'Multi-scale encoder-decoder ONNX model. Takes cloth+mask (4ch) and parse_agnostic+densepose (16ch) at 256x192. Produces 5-scale optical flow for cloth warping + 13-channel semantic segmentation prediction.',
    icon: <Cpu size={14} />,
    color: 'text-red-600 bg-red-50 border-red-200',
  },
  postprocessing: {
    label: 'Cloth Mask Composition',
    description: 'Multiplies the predicted segmap\'s upper-clothes channel (ch 3) by the warped cloth mask. This constrains the clothing region in the segmentation to match where the warped cloth actually is.',
    icon: <Layers size={14} />,
    color: 'text-amber-600 bg-amber-50 border-amber-200',
  },
  warping: {
    label: 'Flow Upsampling & Cloth Warping',
    description: 'Upsamples optical flow from 128x96 to 1024x768 (bicubic + 15px Gaussian smoothing + value clamping). Then warps the full-resolution cloth image using cv2.remap with bicubic interpolation.',
    icon: <Move size={14} />,
    color: 'text-green-600 bg-green-50 border-green-200',
  },
  occlusion: {
    label: 'Occlusion Handling',
    description: 'Upsamples 13-ch segmap to full resolution, applies Gaussian blur (15x15, sigma=3.0), computes softmax. Subtracts body-part regions (face, hair, arms) from the warped cloth mask so cloth doesn\'t overlap body parts. Non-cloth areas composited with white.',
    icon: <Scissors size={14} />,
    color: 'text-pink-600 bg-pink-50 border-pink-200',
  },
};

const DEFAULT_STAGE: StageInfo = {
  label: 'Pipeline Step',
  description: '',
  icon: <Info size={14} />,
  color: 'text-gray-600 bg-gray-50 border-gray-200',
};

const levelBadge = (level: string) => {
  switch (level) {
    case 'warn':
      return <span className="inline-flex items-center gap-0.5 text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded text-[10px] font-medium"><AlertTriangle size={10} /> WARN</span>;
    case 'error':
      return <span className="inline-flex items-center gap-0.5 text-red-700 bg-red-50 px-1.5 py-0.5 rounded text-[10px] font-medium"><AlertCircle size={10} /> ERROR</span>;
    default:
      return null;
  }
};

const LogPanel: React.FC<LogPanelProps> = ({ logs }) => {
  const [expanded, setExpanded] = useState(false);
  const [expandedStages, setExpandedStages] = useState<Set<string>>(new Set());
  const warnCount = logs.filter((l) => l.level === 'warn' || l.level === 'error').length;

  // Group logs by step, preserving order of first appearance
  const groupedLogs: { step: string; entries: PipelineLogEntry[] }[] = [];
  const stepOrder: string[] = [];
  for (const log of logs) {
    if (!stepOrder.includes(log.step)) {
      stepOrder.push(log.step);
      groupedLogs.push({ step: log.step, entries: [] });
    }
    groupedLogs.find(g => g.step === log.step)!.entries.push(log);
  }

  const toggleStage = (step: string) => {
    setExpandedStages(prev => {
      const next = new Set(prev);
      if (next.has(step)) next.delete(step);
      else next.add(step);
      return next;
    });
  };

  const expandAll = () => {
    if (expandedStages.size === groupedLogs.length) {
      setExpandedStages(new Set());
    } else {
      setExpandedStages(new Set(groupedLogs.map(g => g.step)));
    }
  };

  return (
    <div className="mt-3 border border-gray-200 rounded-lg bg-white/80 backdrop-blur shadow-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-gray-50 transition-colors rounded-lg"
      >
        <span className="font-semibold text-gray-700 text-sm">Pipeline Log</span>
        <span className="flex items-center gap-2">
          {warnCount > 0 && (
            <span className="bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full text-xs font-medium">
              {warnCount} warning{warnCount > 1 ? 's' : ''}
            </span>
          )}
          {warnCount === 0 && logs.length > 0 && (
            <span className="bg-green-100 text-green-800 px-2 py-0.5 rounded-full text-xs font-medium">
              All OK
            </span>
          )}
          <span className="text-gray-400 text-xs">{groupedLogs.length} stages</span>
          <ChevronDown
            size={16}
            className={`text-gray-500 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
          />
        </span>
      </button>
      {expanded && (
        <div className="border-t border-gray-200 px-3 py-2">
          <div className="flex justify-end mb-2">
            <button
              onClick={expandAll}
              className="text-xs text-brand-600 hover:text-brand-700 font-medium"
            >
              {expandedStages.size === groupedLogs.length ? 'Collapse all' : 'Expand all'}
            </button>
          </div>
          <div className="space-y-1.5 max-h-[400px] overflow-y-auto">
            {groupedLogs.map(({ step, entries }) => {
              const info = STAGE_INFO[step] || { ...DEFAULT_STAGE, label: step };
              const isOpen = expandedStages.has(step);
              const hasWarnings = entries.some(e => e.level === 'warn' || e.level === 'error');

              return (
                <div key={step} className={`border rounded-lg overflow-hidden ${hasWarnings ? 'border-amber-200' : 'border-gray-150'}`}>
                  <button
                    onClick={() => toggleStage(step)}
                    className={`w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-gray-50 transition-colors ${info.color.split(' ').slice(1).join(' ')}`}
                  >
                    {isOpen ? <ChevronDown size={12} className="shrink-0 text-gray-400" /> : <ChevronRight size={12} className="shrink-0 text-gray-400" />}
                    <span className={`shrink-0 ${info.color.split(' ')[0]}`}>{info.icon}</span>
                    <span className="font-semibold text-xs text-gray-800 flex-1">{info.label}</span>
                    {hasWarnings && <AlertTriangle size={12} className="text-amber-500 shrink-0" />}
                    <span className="text-[10px] text-gray-400">{entries.length} log{entries.length > 1 ? 's' : ''}</span>
                  </button>
                  {isOpen && (
                    <div className="px-3 py-2 bg-white border-t border-gray-100 space-y-1.5">
                      {/* Stage description */}
                      <p className="text-[11px] text-gray-500 italic leading-relaxed mb-2">
                        {info.description}
                      </p>
                      {/* Log entries */}
                      {entries.map((entry, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs">
                          {levelBadge(entry.level)}
                          <span className={`break-words leading-relaxed ${entry.level === 'warn' ? 'text-amber-700' : entry.level === 'error' ? 'text-red-700' : 'text-gray-700'}`}>
                            {entry.message}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default LogPanel;
