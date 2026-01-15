// CLIP Token UMAP Interactive Visualization
let data = [];
let quadtree;
let highlightedPoints = new Set();
let searchResults = new Set();

// Color scales based on token type
const tokenTypeColors = {
  'subword': '#FF8C00',      // Dark orange
  'word_end': '#1E90FF',     // Dodger blue
  'special': '#DC143C',       // Crimson
  'word': '#32CD32',          // Lime green
  'default': '#808080'        // Gray
};

const tokenTypeColorScale = d => {
  const type = d.token_type || 'default';
  return tokenTypeColors[type] || tokenTypeColors['default'];
};

const webglColor = color => {
  const { r, g, b, opacity } = d3.color(color).rgb();
  return [r / 255, g / 255, b / 255, opacity || 1];
};

// Create color function
let currentFillColor = d => {
  if (highlightedPoints.has(d.token_id)) {
    return webglColor('#FF0000'); // Red for highlighted
  }
  if (searchResults.has(d.token_id)) {
    return webglColor('#00FF00'); // Green for search results
  }
  return webglColor(tokenTypeColorScale(d));
};

// Setup scales
let xScale = d3.scaleLinear();
let yScale = d3.scaleLinear();
let xScaleOriginal, yScaleOriginal;

// Create point series
const pointSeries = fc
  .seriesWebglPoint()
  .equals((a, b) => a === b)
  .size(5)
  .crossValue(d => d.x)
  .mainValue(d => d.y);

// Create highlight series (for selection/search) - drawn on top
const highlightSeries = fc
  .seriesWebglPoint()
  .equals((a, b) => a === b)
  .size(150) // Very large for visibility
  .crossValue(d => d.x)
  .mainValue(d => d.y);

// Setup zoom
const zoom = d3
  .zoom()
  .scaleExtent([0.1, 1000])
  .on("zoom", function () {
    const transform = d3.event.transform;
    xScale.domain(transform.rescaleX(xScaleOriginal).domain());
    yScale.domain(transform.rescaleY(yScaleOriginal).domain());
    redraw();
  });


// Pointer handler for hover/click
const pointer = fc.pointer().on("point", ([coord]) => {
  if (!coord || !quadtree) {
    // If we mouse out, clear selection and redraw if needed
    if (highlightedPoints.size > 0) {
      highlightedPoints.clear();
      redraw();
    }
    return;
  }

  const x = xScale.invert(coord.x);
  const y = yScale.invert(coord.y);
  const radius = Math.abs(xScale.invert(coord.x) - xScale.invert(coord.x - 20));
  const closestDatum = quadtree.find(x, y, radius);

  if (closestDatum) {
    if (!highlightedPoints.has(closestDatum.token_id)) {
      highlightedPoints.clear();
      highlightedPoints.add(closestDatum.token_id);
      showTokenInfo(closestDatum);
      redraw();
    }
  } else {
    if (highlightedPoints.size > 0) {
      highlightedPoints.clear();
      redraw();
    }
  }
});

// Annotation series - simplified (no annotations for now)
const annotationSeries = fc.seriesSvgMulti()
  .series([]);

// Create chart
const chart = fc
  .chartCartesian(xScale, yScale)
  .webglPlotArea(
    fc
      .seriesWebglMulti()
      .series([pointSeries, highlightSeries]) // Draw highlightSeries ON TOP
      .mapping((dataObj, index, series) => {
        // dataObj is { data: [...] }
        const data = dataObj.data;

        if (series[index] === pointSeries) {
          return data; // Main series gets all data
        }
        // Highlight series gets only highlighted/searched points
        return data.filter(d => highlightedPoints.has(d.token_id) || searchResults.has(d.token_id));
      })
  )
  .svgPlotArea(
    fc
      .seriesSvgMulti()
      .series([])
      .mapping(d => [])
  )
  .decorate(sel =>
    sel
      .enter()
      .select("d3fc-svg.plot-area")
      .on("measure.range", () => {
        xScaleOriginal.range([0, d3.event.detail.width]);
        yScaleOriginal.range([d3.event.detail.height, 0]);
      })
      .call(zoom)
      .call(pointer)
  );

// Redraw function
// Redraw function
function redraw() {
  // Update fill color for main series
  const mainFillColor = d => {
    return webglColor(tokenTypeColorScale(d));
  };

  const highlightFillColor = d => {
    if (highlightedPoints.has(d.token_id)) {
      return webglColor('#FF0000'); // Red for hovered
    }
    if (searchResults.has(d.token_id)) {
      return webglColor('#00FF00'); // Green for search
    }
    return webglColor('#FFFF00'); // Fallback
  };

  const mainFillColorValue = fc.webglFillColor().value(mainFillColor).data(data);

  pointSeries.decorate(program => {
    mainFillColorValue(program);
  });

  // Highlight series decoration
  const highlightData = data.filter(d => highlightedPoints.has(d.token_id) || searchResults.has(d.token_id));
  const highlightFillColorValue = fc.webglFillColor().value(highlightFillColor).data(highlightData);

  highlightSeries.decorate(program => {
    highlightFillColorValue(program);
  });

  d3.select("#chart").datum({ data }).call(chart);
}

// Show token information
function showTokenInfo(token) {
  const infoPanel = document.getElementById('info-panel');
  const infoContent = document.getElementById('info-content');

  infoContent.innerHTML = `
    <p><strong>Token ID:</strong> ${token.token_id}</p>
    <p><strong>Token:</strong> ${token.token}</p>
    <p><strong>Clean Token:</strong> ${token.clean_token}</p>
    <p><strong>Type:</strong> ${token.token_type}</p>
    <p><strong>Position:</strong> (${token.x.toFixed(2)}, ${token.y.toFixed(2)})</p>
  `;

  infoPanel.classList.add('visible');
}

// Load data
function loadData() {
  d3.csv('umap_visualization.csv')
    .then(function (rows) {
      if (!rows || rows.length === 0) {
        throw new Error('CSV file is empty');
      }

      data = rows.map(d => ({
        token_id: parseInt(d.token_id),
        token: d.token || '',
        clean_token: d.clean_token || d.token || '',
        token_type: d.token_type || 'default',
        x: parseFloat(d.x),
        y: parseFloat(d.y)
      })).filter(d => !isNaN(d.x) && !isNaN(d.y)); // Filter out invalid coordinates


      if (data.length === 0) {
        throw new Error('No valid data points found in CSV');
      }

      // Setup scales based on data
      const xExtent = d3.extent(data, d => d.x);
      const yExtent = d3.extent(data, d => d.y);

      xScale.domain(xExtent);
      yScale.domain(yExtent);
      xScaleOriginal = xScale.copy();
      yScaleOriginal = yScale.copy();

      // Create quadtree for fast spatial search
      quadtree = d3.quadtree()
        .x(d => d.x)
        .y(d => d.y)
        .addAll(data);

      // Hide loading
      document.getElementById('loading').classList.add('hidden');

      // Initial draw
      redraw();

      // Setup search
      setupSearch();

      console.log(`Loaded ${data.length} tokens`);
    })
    .catch(function (error) {
      console.error('Error loading data:', error);
      const loadingEl = document.getElementById('loading');
      loadingEl.innerHTML = `
        <div style="text-align: center; padding: 20px;">
          <p style="color: red; font-size: 16px; margin-bottom: 10px;">Error loading data</p>
          <p style="font-size: 14px; margin-bottom: 10px;">${error.message || 'Unknown error'}</p>
          <p style="font-size: 12px; color: #666;">
            Please ensure:<br>
            1. umap_visualization.csv exists in the same folder<br>
            2. You're running from a local server (not file://)<br>
            3. The CSV is not empty and has valid x/y columns
          </p>
        </div>
      `;
    });
}

// Setup search functionality
function setupSearch() {
  const searchInput = document.getElementById('search-input');
  const searchButton = document.getElementById('search-button');

  function performSearch() {
    const query = searchInput.value.trim();
    if (!query) {
      clearSearch();
      return;
    }

    // Clear previous results
    clearSearch();

    // Parse search terms (separated by comma or space)
    const terms = query.split(/[,\s]+/).filter(t => t.length > 0);

    searchResults.clear();

    terms.forEach(term => {
      term = term.trim();

      // Try as token_id first
      if (/^\d+$/.test(term)) {
        const tokenId = parseInt(term);
        const token = data.find(d => d.token_id === tokenId);
        if (token) {
          searchResults.add(token.token_id);
        }
        return;
      }

      // Exact match on token text (case-insensitive)
      const lowerTerm = term.toLowerCase();
      data.forEach(d => {
        const cleanToken = (d.clean_token || '').toLowerCase();
        const rawToken = (d.token || '').toLowerCase();
        if (cleanToken === lowerTerm || rawToken === lowerTerm) {
          searchResults.add(d.token_id);
        }
      });
    });

    if (searchResults.size > 0) {
      console.log(`Found ${searchResults.size} matching tokens`);
      redraw();
    } else {
      alert(`No tokens found matching: ${query}`);
    }
  }

  function clearSearch() {
    searchResults.clear();
    highlightedPoints.clear();
    searchInput.value = '';
    document.getElementById('info-panel').classList.remove('visible');
    redraw();
  }

  searchButton.addEventListener('click', performSearch);
  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      performSearch();
    }
  });

  // Clear search on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      clearSearch();
    }
  });
}

// Initialize when page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', loadData);
} else {
  loadData();
}
