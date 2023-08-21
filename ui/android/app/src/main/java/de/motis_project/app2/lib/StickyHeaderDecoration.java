/*
 * Copyright 2014 Eduardo Barrenechea
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Source: https://github.com/edubarr/header-decor
// 2016-08-07: modified to not draw a margin
// 2016-10-15: removed cache, consider layout margin of items for header placement
// 2016-10-16: use layout manager to get adapter positions

package de.motis_project.app2.lib;

import android.graphics.Canvas;
import android.graphics.Rect;
import android.util.LongSparseArray;
import android.view.View;
import android.view.ViewGroup;

import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

public class StickyHeaderDecoration extends RecyclerView.ItemDecoration {

    private final StickyHeaderAdapter adapter;
    private final LongSparseArray<RecyclerView.ViewHolder> headerCache = new LongSparseArray<>();

    public StickyHeaderDecoration(StickyHeaderAdapter adapter) {
        this.adapter = adapter;
    }

    @Override
    public void getItemOffsets(
            Rect outRect, View view, RecyclerView parent, RecyclerView.State state) {
        int position = parent.getChildAdapterPosition(view);

        int headerHeight = 0;
        if (position != RecyclerView.NO_POSITION && hasHeader(position)) {
            View header = getHeader(parent, position).itemView;
            headerHeight = header.getHeight();
        }

        outRect.set(0, headerHeight, 0, 0);
    }

    public void clearCache() {
        headerCache.clear();
    }

    private boolean hasHeader(int position) {
        if (position == 0) {
            return true;
        }

        int previous = position - 1;
        return adapter.getHeaderId(position) != adapter.getHeaderId(previous);
    }

    private RecyclerView.ViewHolder getHeader(RecyclerView parent, int position) {
        final long key = adapter.getHeaderId(position);

        RecyclerView.ViewHolder cachedHeader = headerCache.get(key, null);
        if (cachedHeader != null) {
            return cachedHeader;
        } else {
            final RecyclerView.ViewHolder holder = adapter.onCreateHeaderViewHolder(parent);
            final View header = holder.itemView;

            adapter.onBindHeaderViewHolder(holder, position);

            int widthSpec = View.MeasureSpec.makeMeasureSpec(
                    parent.getWidth(), View.MeasureSpec.EXACTLY);
            int heightSpec = View.MeasureSpec.makeMeasureSpec(
                    parent.getHeight(), View.MeasureSpec.UNSPECIFIED);

            int childWidth = ViewGroup.getChildMeasureSpec(
                    widthSpec, 0, header.getLayoutParams().width);
            int childHeight = ViewGroup.getChildMeasureSpec(
                    heightSpec, 0, header.getLayoutParams().height);

            header.measure(childWidth, childHeight);
            header.layout(0, 0, header.getMeasuredWidth(), header.getMeasuredHeight());

            headerCache.put(key, holder);

            return holder;
        }
    }

    int getHeaderYPos(RecyclerView recyclerView, int i) {
        RecyclerView.ViewHolder holder = recyclerView.findViewHolderForAdapterPosition(i);
        if (holder == null) {
            return -1;
        }

        View child = holder.itemView;
        View header = getHeader(recyclerView, i).itemView;

        RecyclerView.LayoutParams params =
                (RecyclerView.LayoutParams) child.getLayoutParams();
        int headerOffset = params.topMargin + header.getHeight();

        return ((int) child.getY()) - headerOffset;
    }

    @Override
    public void onDrawOver(Canvas c, RecyclerView recyclerView, RecyclerView.State state) {
        final LinearLayoutManager layoutManager = (LinearLayoutManager) recyclerView.getLayoutManager();
        final int from = layoutManager.findFirstVisibleItemPosition();
        final int to = layoutManager.findLastVisibleItemPosition();

        if (from == RecyclerView.NO_POSITION || to == RecyclerView.NO_POSITION) {
            return;
        }

        for (int i = from; i <= to; ++i) {
            if (!hasHeader(i) && i != from) {
                continue;
            }

            int top = Math.max(0, getHeaderYPos(recyclerView, i));
            if (i == from) {
                int headerHeight = getHeader(recyclerView, i).itemView.getHeight();
                for (int j = from + 1; j <= to; ++j) {
                    if (!hasHeader(j)) {
                        continue;
                    }

                    int yPos = getHeaderYPos(recyclerView, j);
                    yPos -= headerHeight;
                    if (yPos < 0) {
                        top = yPos;
                    }
                    break;
                }
            }

            c.save();
            c.translate(0, top);
            getHeader(recyclerView, i).itemView.draw(c);
            c.restore();
        }
    }
}
