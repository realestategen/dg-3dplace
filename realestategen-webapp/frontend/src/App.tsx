import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import VideosPage from "./pages/VideosPage";
import SceneViewerPage from "./pages/SceneViewerPage";
import ScenesListPage from "./pages/ScenesListPage";
import "./App.css";

function Nav() {
  const loc = useLocation();
  return (
    <nav className="nav">
      <span className="nav-brand">RealEstateGen · DG-3DGS</span>
      <div className="nav-links">
        <Link className={loc.pathname === "/" ? "active" : ""} to="/">Videos</Link>
        <Link className={loc.pathname.startsWith("/scenes") ? "active" : ""} to="/scenes">Scenes</Link>
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Nav />
      <main className="main">
        <Routes>
          <Route path="/" element={<VideosPage />} />
          <Route path="/scenes" element={<ScenesListPage />} />
          <Route path="/scenes/:id" element={<SceneViewerPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
