package de.motis_project.app2.journey;

import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

public class InfiniteScroll extends RecyclerView.OnScrollListener {

    public interface Loader {
        void loadBefore();
        void loadAfter();
    }

    private final LinearLayoutManager layoutManager;
    private final JourneySummaryAdapter adapter;
    private final Loader loader;

    private boolean loadingBefore = false;
    private boolean loadingAfter = false;

    InfiniteScroll(Loader loader, LinearLayoutManager layoutManager, JourneySummaryAdapter adapter) {
        this.loader = loader;
        this.layoutManager = layoutManager;
        this.adapter = adapter;
    }

    @Override
    public void onScrolled(RecyclerView recyclerView, int dx, int dy) {
        if (dy != 0) {
            onScrolled();
        }
    }

    public void onScrolled() {
        onScrolled(layoutManager.findFirstVisibleItemPosition());
    }

    private void onScrolled(int first) {
        synchronized (layoutManager) {
            if (!loadingAfter) {
                int last = layoutManager.findLastVisibleItemPosition();
                if (last != RecyclerView.NO_POSITION && adapter.getItemViewType(last) == JourneySummaryAdapter.VIEW_TYPE_LOADING_SPINNER) {
                    loadingAfter = true;
                    loader.loadAfter();
                    return;
                }
            }
            if (!loadingBefore) {
                if (first == 0) {
                    loadingBefore = true;
                    loader.loadBefore();
                    return;
                }
            }
        }
    }

    public void notifyLoadBeforeFinished(int firstVisible) {
        loadingBefore = false;
        onScrolled(firstVisible);
    }

    public void notifyLoadBeforeFinished() {
        loadingBefore = false;
        onScrolled();
    }

    public void notifyLoadAfterFinished() {
        loadingAfter = false;
        onScrolled();
    }

    public void notifyLoadFinished() {
        notifyLoadAfterFinished();
        notifyLoadBeforeFinished();
    }

    public void setLoading() {
        loadingAfter = true;
        loadingBefore = true;
    }
}
