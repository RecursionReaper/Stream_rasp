/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/py/:path*',
        destination: 'http://192.168.1.13:8000/:path*',
      },
    ];
  },
};

export default nextConfig;
