function handleGraphClick(event, graphContainerId, triggerId) {
    try {
        var graphContainer = document.getElementById(graphContainerId);
        if (!graphContainer) {
            return {x: 0.0, y: 0.0, triggerIdPassed: triggerId, graphFound: "no", insideGraph: "no", foundPoint: "no"};
        }

        var graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const clickX = event.clientX;
        const clickY = event.clientY;

        if (clickX >= bbox.left && clickX <= bbox.right &&
            clickY >= bbox.top && clickY <= bbox.bottom) {

            const layout = graph._fullLayout;
            const xaxis = layout.xaxis;
            const yaxis = layout.yaxis;

            const xPixel = clickX - bbox.left;
            const yPixel = clickY - bbox.top;

            const plotX0 = xaxis._offset;
            const plotX1 = xaxis._offset + xaxis._length;
            const plotY0 = yaxis._offset;
            const plotY1 = yaxis._offset + yaxis._length;

            if (xPixel >= plotX0 && xPixel <= plotX1 &&
                yPixel >= plotY0 && yPixel <= plotY1) {

                const xData = xaxis.p2d(xPixel - xaxis._offset);
                const yData = yaxis.p2d(yPixel - yaxis._offset);

                const PIXEL_RADIUS = 4;

                let foundPoint = false;
                for (const trace of graph.data) {
                    if (!trace.x || !trace.y) continue;
                    for (let i = 0; i < trace.x.length; i++) {
                        const xIterGraphPixel = xaxis.d2p(trace.x[i]) + xaxis._offset;
                        const yIterGraphPixel = yaxis.d2p(trace.y[i]) + yaxis._offset;
                        if (Math.abs(xIterGraphPixel - xPixel) < PIXEL_RADIUS &&
                        Math.abs(yIterGraphPixel - yPixel) < PIXEL_RADIUS) {
                            foundPoint = true;
                            break;
                        }
                    }
                    if (foundPoint) break;
                }

                if (!foundPoint) {
                    console.log("Clicked on empty area:", xData, yData);
                    return {x: xData, y: yData, triggerIdPassed: triggerId, graphFound: "yes", insideGraph: "yes", foundPoint: "yes"}
                }
                else {
                    return {x: xData, y: yData, triggerIdPassed: triggerId, graphFound: "yes", insideGraph: "yes", foundPoint: "no"}
                }
            }
            else {
                return {x: 0.0, y: 0.0, triggerIdPassed: triggerId, graphFound: "yes", insideGraph: "no", foundPoint: "no"}
            }
        }
        else {
            return {x: 0.0, y: 0.0, triggerIdPassed: triggerId, graphFound: "yes", insideGraph: "no", foundPoint: "no"}
        }
    } catch (e) {
        console.error("handleGraphClick error:", e);
        return null;
    }
}


let ignoreNextClick = false;

window.addEventListener("click", function(event) {
    if (ignoreNextClick) {
        ignoreNextClick = false;
        return;
    }

    if (
        event.target.closest('.Select-menu') ||
        event.target.closest('.Select-control') ||
        event.target.closest('.dropdown-container') ||
        event.target.tagName === "SELECT"
    ) {
        return;
    }

    window.lastClickEvent = {
      clientX: event.clientX,
      clientY: event.clientY,
      altKey: event.altKey,
      ctrlKey: event.ctrlKey,
      shiftKey: event.shiftKey,
      timeStamp: event.timeStamp
    };

    const configs = [
        { dashboard: "euclidean", dataset: "MNIST", projection: "TriMap", trigger: "euclidean-mnist-trimap-trigger" },
        { dashboard: "euclidean", dataset: "FashionMNIST", projection: "TriMap", trigger: "euclidean-fashion-mnist-trimap-trigger" },
        { dashboard: "euclidean", dataset: "CIFAR-100", projection: "TriMap", trigger: "euclidean-cifar-trimap-trigger" },
        { dashboard: "euclidean", dataset: "MNIST", projection: "UMAP", trigger: "euclidean-mnist-umap-trigger" },
        { dashboard: "euclidean", dataset: "FashionMNIST", projection: "UMAP", trigger: "euclidean-fashion-mnist-umap-trigger" },
        { dashboard: "euclidean", dataset: "CIFAR-100", projection: "UMAP", trigger: "euclidean-cifar-umap-trigger" },
        { dashboard: "euclidean", dataset: "MNIST", projection: "t_SNE", trigger: "euclidean-mnist-tsne-trigger" },
        { dashboard: "euclidean", dataset: "FashionMNIST", projection: "t_SNE", trigger: "euclidean-fashion-mnist-tsne-trigger" },
        { dashboard: "euclidean", dataset: "CIFAR-100", projection: "t_SNE", trigger: "euclidean-cifar-tsne-trigger" },
        { dashboard: "euclidean", dataset: "MNIST", projection: "PCA", trigger: "euclidean-mnist-pca-trigger" },
        { dashboard: "euclidean", dataset: "FashionMNIST", projection: "PCA", trigger: "euclidean-fashion-mnist-pca-trigger" },
        { dashboard: "euclidean", dataset: "CIFAR-100", projection: "PCA", trigger: "euclidean-cifar-pca-trigger" },
        "model-prog-plot"
    ];

    let searchResult = null;
    let foundGraphId = null;
    for (const config of configs) {
        let result;
        if (config === "model-prog-plot") {
            result = handleGraphClick(
                event,
                "model-prog-plot",
                "model-prog-trimap-trigger"
            );
        }
        else {
            result = handleGraphClick(
                event,
                JSON.stringify({ dashboard: config.dashboard, dataset: config.dataset, projection: config.projection, type: "scatterplot" }),
                config.trigger
            );
        }

        if (result.graphFound === "yes") {
            if (result.insideGraph === "yes") {
                searchResult = result;
                foundGraphId = config;
                break;
            }
        }
    }
    
    if (searchResult) {
        console.log(searchResult);
        if (searchResult.foundPoint === "yes") {
            const button = document.getElementById(searchResult.triggerIdPassed);
            if (button) {
                console.log("Clicking Trigger");
                ignoreNextClick = true;
                button.click();
            }
        }
    }
});
