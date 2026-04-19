import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';

export default function CursorEffect() {
  const cursorRef = useRef<HTMLDivElement>(null);
  const cursorDotRef = useRef<HTMLDivElement>(null);
  const [isHovering, setIsHovering] = useState(false);
  const [cursorColor, setCursorColor] = useState('#22D3EE');

  useEffect(() => {
    // Skip on touch devices
    if (window.matchMedia('(pointer: coarse)').matches) return;

    const cursor = cursorRef.current;
    const cursorDot = cursorDotRef.current;
    if (!cursor || !cursorDot) return;

    const moveCursor = (e: MouseEvent) => {
      // Fast, responsive cursor movement
      gsap.to(cursor, {
        x: e.clientX,
        y: e.clientY,
        duration: 0.08,
        ease: 'power2.out',
      });
      gsap.to(cursorDot, {
        x: e.clientX,
        y: e.clientY,
        duration: 0.02,
        ease: 'none',
      });
    };

    // Detect hoverable elements
    const handleMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      
      if (
        target.tagName === 'BUTTON' ||
        target.tagName === 'A' ||
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.closest('button') ||
        target.closest('a') ||
        target.closest('[role="button"]') ||
        target.classList.contains('cursor-pointer')
      ) {
        setIsHovering(true);
        
        // Color detection
        if (target.closest('[data-cursor-color]')) {
          const colorEl = target.closest('[data-cursor-color]') as HTMLElement;
          setCursorColor(colorEl.dataset.cursorColor || '#22D3EE');
        } else if (target.closest('.severity-severe, .text-red-500, [data-cursor="red"]')) {
          setCursorColor('#EF4444');
        } else if (target.closest('.severity-moderate, .text-orange-500, [data-cursor="orange"]')) {
          setCursorColor('#F97316');
        } else if (target.closest('.severity-mild, .text-amber-500, [data-cursor="amber"]')) {
          setCursorColor('#F59E0B');
        } else {
          setCursorColor('#22D3EE');
        }
      }
    };

    const handleMouseOut = () => {
      setIsHovering(false);
      setCursorColor('#22D3EE');
    };

    window.addEventListener('mousemove', moveCursor);
    document.addEventListener('mouseover', handleMouseOver);
    document.addEventListener('mouseout', handleMouseOut);

    return () => {
      window.removeEventListener('mousemove', moveCursor);
      document.removeEventListener('mouseover', handleMouseOver);
      document.removeEventListener('mouseout', handleMouseOut);
    };
  }, []);

  // Don't render on touch devices
  if (typeof window !== 'undefined' && window.matchMedia('(pointer: coarse)').matches) {
    return null;
  }

  return (
    <>
      {/* Outer ring */}
      <div
        ref={cursorRef}
        className="fixed top-0 left-0 pointer-events-none z-[9999] -translate-x-1/2 -translate-y-1/2 mix-blend-difference hidden lg:block"
        style={{ willChange: 'transform' }}
      >
        <div
          className="rounded-full border-2 transition-all duration-200"
          style={{
            width: isHovering ? 48 : 32,
            height: isHovering ? 48 : 32,
            borderColor: cursorColor,
            backgroundColor: isHovering ? `${cursorColor}15` : 'transparent',
            boxShadow: `0 0 ${isHovering ? 20 : 10}px ${cursorColor}40`,
          }}
        />
      </div>

      {/* Center dot */}
      <div
        ref={cursorDotRef}
        className="fixed top-0 left-0 pointer-events-none z-[9999] -translate-x-1/2 -translate-y-1/2 hidden lg:block"
        style={{ willChange: 'transform' }}
      >
        <div
          className="w-1.5 h-1.5 rounded-full"
          style={{
            backgroundColor: cursorColor,
            boxShadow: `0 0 6px ${cursorColor}`,
          }}
        />
      </div>

      {/* Hide default cursor on desktop */}
      <style>{`
        @media (pointer: fine) {
          * { cursor: none !important; }
        }
      `}</style>
    </>
  );
}
