import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, AlertTriangle, CheckCircle } from 'lucide-react';
import clsx from 'clsx';

interface ImageUploaderProps {
  label: string;
  description?: string;
  guidance?: string;
  onImageSelect: (file: File) => void;
  previewUrl?: string;
  onClear?: () => void;
  disabled?: boolean;
  disabledMessage?: string;
}

interface ImageInfo {
  width: number;
  height: number;
  sizeKB: number;
  quality: 'good' | 'warning' | 'error';
  message?: string;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({
  label,
  description,
  guidance,
  onImageSelect,
  previewUrl,
  onClear,
  disabled = false,
  disabledMessage,
}) => {
  const [imageInfo, setImageInfo] = useState<ImageInfo | null>(null);

  const validateImage = useCallback((file: File): Promise<ImageInfo> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const sizeKB = Math.round(file.size / 1024);
        let quality: ImageInfo['quality'] = 'good';
        let message: string | undefined;

        if (img.width < 256 || img.height < 256) {
          quality = 'warning';
          message = 'Low resolution - results may be blurry';
        } else if (img.width > 4096 || img.height > 4096) {
          quality = 'warning';
          message = 'Very large image - will be downscaled';
        } else if (img.width / img.height > 3 || img.height / img.width > 3) {
          quality = 'warning';
          message = 'Unusual aspect ratio - consider cropping';
        }

        resolve({ width: img.width, height: img.height, sizeKB, quality, message });
        URL.revokeObjectURL(img.src);
      };
      img.onerror = () => {
        resolve({ width: 0, height: 0, sizeKB: Math.round(file.size / 1024), quality: 'error', message: 'Could not read image' });
      };
      img.src = URL.createObjectURL(file);
    });
  }, []);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0 && !disabled) {
        const file = acceptedFiles[0];
        const info = await validateImage(file);
        setImageInfo(info);
        onImageSelect(file);
      }
    },
    [onImageSelect, disabled, validateImage]
  );

  const handleClear = () => {
    setImageInfo(null);
    onClear?.();
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp'],
    },
    maxSize: 10 * 1024 * 1024,
    disabled,
    multiple: false,
  });

  return (
    <div className="w-full">
      <label className="block text-lg font-semibold text-gray-800 mb-1">
        {label}
      </label>
      {description && (
        <p className="text-sm text-gray-600 mb-1">{description}</p>
      )}
      {guidance && !previewUrl && (
        <p className="text-xs text-brand-600 mb-3 italic">{guidance}</p>
      )}

      {previewUrl ? (
        <div className="relative w-full min-h-[250px] max-h-[450px] rounded-lg overflow-hidden shadow-lg border-2 border-brand-200">
          <img
            src={previewUrl}
            alt="Preview"
            className="w-full h-full object-contain bg-gray-50"
            style={{ maxHeight: '450px' }}
          />
          {!disabled && (
            <button
              onClick={handleClear}
              className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors shadow-lg"
              aria-label="Clear image"
            >
              <X size={20} />
            </button>
          )}
          {/* Image info badge */}
          {imageInfo && (
            <div className={clsx(
              'absolute bottom-2 left-2 flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium shadow',
              imageInfo.quality === 'good' && 'bg-green-100 text-green-800',
              imageInfo.quality === 'warning' && 'bg-amber-100 text-amber-800',
              imageInfo.quality === 'error' && 'bg-red-100 text-red-800',
            )}>
              {imageInfo.quality === 'good' ? (
                <CheckCircle size={12} />
              ) : (
                <AlertTriangle size={12} />
              )}
              {imageInfo.width}x{imageInfo.height} &middot; {imageInfo.sizeKB > 1024 ? `${(imageInfo.sizeKB / 1024).toFixed(1)}MB` : `${imageInfo.sizeKB}KB`}
              {imageInfo.message && (
                <span className="ml-1">- {imageInfo.message}</span>
              )}
            </div>
          )}
        </div>
      ) : (
        <div
          {...getRootProps()}
          className={clsx(
            'w-full min-h-[250px] max-h-[450px] h-[40vh] border-2 border-dashed rounded-lg flex flex-col items-center justify-center cursor-pointer transition-all duration-200',
            isDragActive
              ? 'border-brand-500 bg-brand-50'
              : 'border-gray-300 bg-white hover:border-brand-400',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center justify-center">
            <Upload
              size={48}
              className={clsx(
                'mb-4',
                isDragActive ? 'text-brand-500' : 'text-gray-400'
              )}
            />
            <p className="text-lg font-medium text-gray-700 text-center px-4">
              {isDragActive
                ? 'Drop your image here'
                : 'Drag and drop your image here'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              or click to select a file
            </p>
            <p className="text-xs text-gray-400 mt-3">
              JPG, PNG, WebP &middot; Max 10MB &middot; Min 256x256
            </p>
            {disabledMessage && (
              <p className="text-xs text-red-400 mt-2 font-medium">{disabledMessage}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
