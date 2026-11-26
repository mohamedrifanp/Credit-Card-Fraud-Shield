export function ShieldLogo({ className = 'w-9 h-9' }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      {/* Outer protective shield */}
      <path
        d="M20 3L6 8.5V19.2C6 27.9 11.9 35.7 20 38C28.1 35.7 34 27.9 34 19.2V8.5L20 3Z"
        fill="#0F172A"
        stroke="#38BDF8"
        strokeWidth="1.75"
        strokeLinejoin="round"
      />
      {/* Credit card silhouette inside shield */}
      <rect
        x="11"
        y="14"
        width="18"
        height="11.5"
        rx="2"
        fill="#1E293B"
        stroke="#0EA5E9"
        strokeWidth="1.4"
      />
      {/* Magnetic stripe */}
      <rect x="11" y="16.8" width="18" height="2.4" fill="#38BDF8" fillOpacity="0.45" />
      {/* Card chip */}
      <rect x="13.2" y="21" width="3.5" height="2.4" rx="0.6" fill="#38BDF8" />
      {/* Security checkmark badge at bottom-right of shield */}
      <circle cx="28" cy="26.5" r="5.5" fill="#0B1220" stroke="#10B981" strokeWidth="1.5" />
      <path
        d="M25.6 26.5L27.2 28.1L30.5 24.8"
        stroke="#10B981"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
