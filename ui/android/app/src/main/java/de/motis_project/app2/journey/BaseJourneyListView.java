package de.motis_project.app2.journey;

import android.content.Context;
import android.content.res.Resources;
import android.util.AttributeSet;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;

import de.motis_project.app2.R;
import de.motis_project.app2.io.error.MotisErrorException;
import de.motis_project.app2.lib.SimpleDividerItemDecoration;
import de.motis_project.app2.lib.StickyHeaderDecoration;
import motis.routing.RoutingResponse;

public class BaseJourneyListView<Q>
        extends RecyclerView
        implements InfiniteScroll.Loader {
    private static final String TAG = "BaseJourneyListView";

    protected int STICKY_HEADER_SCROLL_OFFSET;

    protected ConnectionLoader<Q> connectionLoader;
    protected final LinearLayoutManager layoutManager = new LinearLayoutManager(getContext());
    protected JourneySummaryAdapter adapter;
    protected InfiniteScroll infiniteScroll;
    protected StickyHeaderDecoration stickyHeaderDecorator;
    protected LoadResultListener loadResultListener;

    public BaseJourneyListView(Context context) {
        super(context);
    }

    public BaseJourneyListView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public BaseJourneyListView(Context context, @Nullable AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
    }

    public void init(ConnectionLoader<Q> connectionLoader) {
        this.connectionLoader = connectionLoader;

        adapter = new JourneySummaryAdapter(connectionLoader.getData());
        infiniteScroll = new InfiniteScroll(this, layoutManager, adapter);
        stickyHeaderDecorator = new StickyHeaderDecoration(adapter);

        setAdapter(adapter);
        addOnScrollListener(infiniteScroll);
        addItemDecoration(new SimpleDividerItemDecoration(getContext()));
        addItemDecoration(stickyHeaderDecorator);
        setLayoutManager(layoutManager);

        Resources r = getResources();
        STICKY_HEADER_SCROLL_OFFSET = r.getDimensionPixelOffset(R.dimen.journey_list_floating_header_height);
    }

    public LoadResultListener getLoadResultListener() {
        return loadResultListener;
    }

    public void setLoadResultListener(LoadResultListener loadResultListener) {
        this.loadResultListener = loadResultListener;
    }

    public void loadInitial() {
        Log.d(TAG, "loadInitial");
        connectionLoader.reset();

        adapter.setLoadAfterError(null);
        adapter.setLoadBeforeError(null);
        adapter.recalculateHeaders();
        stickyHeaderDecorator.clearCache();
        adapter.notifyDataSetChanged();

        infiniteScroll.setLoading();
        if (loadResultListener != null) {
            loadResultListener.loading(LoadType.Initial);
        }

        connectionLoader.loadInitial((res, newStartIndex, newConnectionCount) -> {
            Log.d(TAG, "loadInitial succeeded");
            adapter.recalculateHeaders();
            stickyHeaderDecorator.clearCache();
            adapter.notifyDataSetChanged();
            layoutManager.scrollToPositionWithOffset(1, STICKY_HEADER_SCROLL_OFFSET);
            infiniteScroll.notifyLoadFinished();
            if (loadResultListener != null) {
                loadResultListener.loaded(LoadType.Initial, res, newConnectionCount);
            }
        }, t -> {
            Log.d(TAG, "loadInitial failed");
            infiniteScroll.notifyLoadFinished();
            if (loadResultListener != null) {
                loadResultListener.failed(LoadType.Initial, t);
            }
        });
    }

    public void notifyDestroy() {
        connectionLoader.destroy();
    }

    @Override
    public void loadBefore() {
        if (adapter.getErrorStateBefore() != JourneySummaryAdapter.ERROR_TYPE_NO_ERROR) {
            return;
        }

        if (loadResultListener != null) {
            loadResultListener.loading(LoadType.Before);
        }

        connectionLoader.loadBefore((res, newStartIndex, newConnectionCount) -> {
            adapter.recalculateHeaders();
            stickyHeaderDecorator.clearCache();
            adapter.notifyItemRangeInserted(1, newConnectionCount);
            if (layoutManager.findFirstVisibleItemPosition() == 0) {
                layoutManager.scrollToPosition(newConnectionCount + 1);
            }

            infiniteScroll.notifyLoadBeforeFinished(newConnectionCount);
            if (res.connectionsLength() == 0) {
                infiniteScroll.onScrolled();
            }
            if (loadResultListener != null) {
                loadResultListener.loaded(LoadType.Before, res, newConnectionCount);
            }
        }, t -> {
            infiniteScroll.notifyLoadBeforeFinished();
            if (t instanceof MotisErrorException) {
                adapter.setLoadBeforeError((MotisErrorException) t);
            }
            if (loadResultListener != null) {
                loadResultListener.failed(LoadType.Before, t);
            }
        });
    }

    @Override
    public void loadAfter() {
        if (adapter.getErrorStateAfter() != JourneySummaryAdapter.ERROR_TYPE_NO_ERROR) {
            return;
        }

        if (loadResultListener != null) {
            loadResultListener.loading(LoadType.After);
        }

        connectionLoader.loadAfter((res, newStartIndex, newConnectionCount) -> {
            adapter.notifyItemRangeInserted(newStartIndex + 1, newConnectionCount);
            adapter.recalculateHeaders();
            stickyHeaderDecorator.clearCache();
            infiniteScroll.notifyLoadAfterFinished();
            if (res.connectionsLength() == 0) {
                infiniteScroll.onScrolled();
            }
            if (loadResultListener != null) {
                loadResultListener.loaded(LoadType.After, res, newConnectionCount);
            }
        }, t -> {
            infiniteScroll.notifyLoadAfterFinished();
            if (t instanceof MotisErrorException) {
                adapter.setLoadAfterError((MotisErrorException) t);
            }
            if (loadResultListener != null) {
                loadResultListener.failed(LoadType.After, t);
            }
        });
    }

    public interface LoadResultListener {
        public void loading(LoadType type);

        public void loaded(LoadType type, RoutingResponse res, int newConnectionCount);

        public void failed(LoadType type, Throwable t);
    }

    public enum LoadType {
        Initial, Before, After
    }

    public List<ConnectionWrapper> getAllConnections() {
        return connectionLoader.getData();
    }

    public List<ConnectionWrapper> getVisibleConnections() {
        ArrayList<ConnectionWrapper> conns = new ArrayList<>();
        int first = layoutManager.findFirstVisibleItemPosition();
        int last = layoutManager.findLastVisibleItemPosition();
//        Log.i(TAG, "getVisibleConnections: first=" + first + ", last=" + last);
        if (first != RecyclerView.NO_POSITION && last != RecyclerView.NO_POSITION) {
            for (int i = first; i <= last; i++) {
                if (adapter.getItemViewType(i) == JourneySummaryAdapter.VIEW_TYPE_JOURNEY_PREVIEW) {
                    ConnectionWrapper con = adapter.getConnection(i);
                    if (con != null) {
                        conns.add(con);
                    }
                }
            }
        }
        return conns;
    }
}
