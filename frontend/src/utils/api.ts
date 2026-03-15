import axios, { AxiosError } from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8020';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 min - generator inference is slow on CPU
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface TryOnInputs {
  person: File;
  cloth: File;
}

export interface ApiError {
  message: string;
  status?: number;
  details?: string;
}

export interface PipelineLogEntry {
  step: string;
  level: 'info' | 'warn' | 'error';
  message: string;
}

export interface PipelinePreview {
  cloth_mask: string;
  agnostic: string;
  warped_cloth: string;
  logs?: PipelineLogEntry[];
}

export async function performTryOn(inputs: TryOnInputs): Promise<string> {
  try {
    const formData = new FormData();
    formData.append('person', inputs.person);
    formData.append('cloth', inputs.cloth);

    const response = await apiClient.post('/tryon-simple', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      responseType: 'blob',
    });

    return URL.createObjectURL(response.data);
  } catch (error) {
    throw handleApiError(error);
  }
}

export async function fetchPipelinePreview(person: File, cloth: File): Promise<PipelinePreview> {
  try {
    const formData = new FormData();
    formData.append('person', person);
    formData.append('cloth', cloth);

    const response = await apiClient.post('/pipeline-preview', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data as PipelinePreview;
  } catch (error) {
    throw handleApiError(error);
  }
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
}

export function getApiErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    if (error.response?.data?.message) {
      return error.response.data.message;
    }
    // Try to read detail from blob error response
    if (error.response?.data?.detail) {
      return error.response.data.detail;
    }
    if (error.response?.status === 408 || error.code === 'ECONNABORTED') {
      return 'Request timed out. The model inference takes ~60s on CPU. Please try again.';
    }
    if (error.response?.status === 400) {
      return 'Invalid request. Please check your uploaded images.';
    }
    if (error.response?.status === 503) {
      return 'Models not loaded. Please check the backend server.';
    }
    if (error.response?.status === 500) {
      return 'Server error during inference. Check backend logs.';
    }
    if (error.message === 'Network Error') {
      return 'Network error. Please check your connection and API server.';
    }
    return error.message || 'An error occurred';
  }

  if (error instanceof Error) {
    return error.message;
  }

  return 'An unknown error occurred';
}

function handleApiError(error: unknown): ApiError {
  const message = getApiErrorMessage(error);
  const status = error instanceof AxiosError ? error.response?.status : undefined;

  return {
    message,
    status,
    details: error instanceof AxiosError ? error.response?.data?.details : undefined,
  };
}

export default apiClient;
