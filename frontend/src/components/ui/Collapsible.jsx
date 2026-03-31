import React, { useState } from 'react';

export const Collapsible = ({ title, children, defaultOpen = true }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className={`section ${isOpen ? 'open' : ''}`}>
      <div className="sec-head" onClick={() => setIsOpen(!isOpen)}>
        <span className="sec-title">{title}</span>
        <span className="sec-caret">›</span>
      </div>
      <div className="sec-body">
        {children}
      </div>
    </div>
  );
};
