import './LoadingIndicator.css';

interface LoadingIndicatorProps {
  message?: string;
}

export const LoadingIndicator = ({ message = 'Loading model...' }: LoadingIndicatorProps) => {
  return (
    <div className="loading-container">
      <div className="loading-spinner">
        <div className="spinner-dot"></div>
        <div className="spinner-dot"></div>
        <div className="spinner-dot"></div>
      </div>
      <p className="loading-message">{message}</p>
    </div>
  );
};
