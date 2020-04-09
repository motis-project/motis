var RailViz = RailViz || {};

RailViz.Textures = (function() {

  function createTrain() {
    return function(size) {
      const border = (2 / 64) * size;
      const padding = (size - (size / 2)) / 2 + border;
      const innerSize = size - (2 * padding);
      const mid = size / 2;
      const rad = innerSize / 3.5;
      var cv = document.createElement('canvas');
      cv.width = size;
      cv.height = size;
      var ctx = cv.getContext('2d', {alpha: true});

      ctx.beginPath();

      ctx.arc(padding + rad, mid, rad, 1 / 2 * Math.PI, 3 / 2 * Math.PI, false);

      ctx.bezierCurveTo(
          padding + rad + rad, mid - rad, size - padding, mid, size - padding,
          mid);
      ctx.bezierCurveTo(
          size - padding, mid, padding + rad + rad, mid + rad, padding + rad,
          mid + rad);

      ctx.closePath();

      ctx.fillStyle = 'rgba(255, 255, 255, 1.0)';
      ctx.fill();
      ctx.lineWidth = border;
      ctx.strokeStyle = 'rgba(160, 160, 160, 1.0)';
      ctx.stroke();
      return cv;
    };
  }

  return {createTrain: createTrain};
})();
