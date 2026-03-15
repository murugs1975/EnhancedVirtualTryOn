import type { AppProps } from 'next/app';
import { Toaster } from 'react-hot-toast';
import '@/styles/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Component {...pageProps} />
      <Toaster
        position="top-right"
        reverseOrder={false}
        gutter={8}
        toastOptions={{
          duration: 4000,
          style: {
            background: '#fff',
            color: '#333',
            boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
          },
          success: {
            style: {
              background: '#10b981',
              color: '#fff',
            },
            icon: '✓',
          },
          error: {
            style: {
              background: '#ef4444',
              color: '#fff',
            },
            icon: '✕',
          },
        }}
      />
    </>
  );
}
