"use client";
import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function Home() {
  const router = useRouter();
  const canvasRef = useRef(null);

  const captureImage = () => {
    const canvas = canvasRef.current;
    const video = document.getElementById("live-stream");

    if (canvas && video) {
      const context = canvas.getContext("2d");
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const imageData = canvas.toDataURL("image/png");
      localStorage.setItem("capturedImage", imageData);

      router.push("/snaps");
    }
  };

  return (
    <div className="grid grid-rows-[auto_auto_1fr_auto] h-screen bg-black text-white font-sans antialiased overflow-hidden">
      <header className="p-4 sm:p-6 text-center relative z-10 shrink-0">
        <span className="text-xl sm:text-2xl md:text-3xl font-bold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-gradient-x">
          Logo
        </span>
      </header>

      <nav className="p-4 sm:p-6 relative z-10 shrink-0">
        <ul className="flex flex-col sm:flex-row justify-center items-center space-y-4 sm:space-y-0 sm:space-x-6 md:space-x-10 text-zinc-300 text-xs sm:text-sm md:text-base font-medium">
          {["Home", "Snaps", "Alerts", "Settings", "About"].map((page) => (
            <li key={page} className="group w-full sm:w-auto px-4 py-2 rounded-full cursor-pointer bg-zinc-900/20 backdrop-blur-md border border-zinc-800/50 hover:bg-gradient-to-r hover:from-indigo-500/20 hover:to-purple-500/20 transition-all duration-300 ease-in-out">
              <Link href={`/${page.toLowerCase()}`} className="group-hover:text-white transition-colors duration-300">{page}</Link>
            </li>
          ))}
        </ul>
      </nav>

      {/* LIVE STREAM SECTION */}
      <main className="flex justify-center items-center flex-1 p-4 sm:p-6 md:p-8 relative z-10 overflow-hidden">
        <div className="bg-gradient-to-br from-zinc-950/90 to-black/90 w-full sm:w-[90%] md:w-[80%] lg:w-[75%] h-full shadow-2xl rounded-xl sm:rounded-2xl flex justify-center items-center border border-zinc-800/30 backdrop-blur-lg overflow-hidden relative">
          <img
            id="live-stream"
            src="http://localhost:8000"
            alt="Live Stream"
            className="w-full h-full object-cover rounded-xl"
          />
          <canvas ref={canvasRef} className="hidden" />
        </div>

        <button
          onClick={captureImage}
          className="mt-6 px-6 py-3 bg-indigo-500 hover:bg-indigo-700 text-white rounded-lg transition-all duration-300"
        >Capture</button>
      </main>

      <footer className="p-4 sm:p-6 text-center text-zinc-400 text-xs sm:text-sm md:text-base relative z-10 bg-gradient-to-t from-zinc-950/80 to-transparent shrink-0">
        Made by some crackheads
      </footer>
    </div>
  );
}
