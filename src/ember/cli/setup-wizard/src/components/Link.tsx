import React from 'react';

interface LinkProps {
  url: string;
  children: React.ReactNode;
}

/**
 * Simple link component that renders children as-is.
 * Modern terminals auto-detect URLs and make them clickable.
 * This avoids deprecated defaultProps warnings from ink-link.
 * 
 * Following KISS principle - terminals handle URL detection better
 * than trying to force ANSI escape sequences.
 */
export const Link: React.FC<LinkProps> = ({url, children}) => {
  // Simply render children - let the terminal handle URL detection
  return <>{children}</>;
};