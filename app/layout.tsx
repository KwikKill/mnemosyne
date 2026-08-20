import type { Metadata, Viewport } from "next"
import { IBM_Plex_Mono } from "next/font/google"
import "./globals.css"

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-mono",
})

export const metadata: Metadata = {
  title: "MNEMOSYNE // Neural Diagnostic Terminal",
  description:
    "A hand-written multilayer perceptron, trained live and inspected in real time through a station diagnostic terminal.",
}

export const viewport: Viewport = {
  colorScheme: "dark",
  themeColor: "#090c0d",
  width: "device-width",
  initialScale: 1,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`dark ${plexMono.variable}`}>
      <body className="bg-background text-foreground font-mono antialiased">{children}</body>
    </html>
  )
}
