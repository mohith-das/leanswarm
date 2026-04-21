import "./globals.css";

export const metadata = {
  title: "Lean Swarm World Inspector",
  description: "Inspect pasted Lean Swarm simulation JSON and explore the resulting world.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
