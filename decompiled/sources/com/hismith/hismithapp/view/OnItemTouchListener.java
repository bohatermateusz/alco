package com.hismith.hismithapp.view;

import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import androidx.core.view.GestureDetectorCompat;
import androidx.recyclerview.widget.RecyclerView;

/* loaded from: classes3.dex */
public abstract class OnItemTouchListener implements RecyclerView.OnItemTouchListener {
    private GestureDetectorCompat mGestureDetectorCompat;
    private RecyclerView mRecyclerView;

    private class MyGestureListener extends GestureDetector.SimpleOnGestureListener {
        private MyGestureListener() {
        }

        @Override // android.view.GestureDetector.SimpleOnGestureListener, android.view.GestureDetector.OnGestureListener
        public void onLongPress(MotionEvent motionEvent) {
            View findChildViewUnder = OnItemTouchListener.this.mRecyclerView.findChildViewUnder(motionEvent.getX(), motionEvent.getY());
            if (findChildViewUnder != null) {
                OnItemTouchListener.this.onItemClick(OnItemTouchListener.this.mRecyclerView.getChildViewHolder(findChildViewUnder));
            }
        }

        @Override // android.view.GestureDetector.SimpleOnGestureListener, android.view.GestureDetector.OnGestureListener
        public boolean onSingleTapUp(MotionEvent motionEvent) {
            View findChildViewUnder = OnItemTouchListener.this.mRecyclerView.findChildViewUnder(motionEvent.getX(), motionEvent.getY());
            if (findChildViewUnder == null) {
                return true;
            }
            OnItemTouchListener.this.onItemClick(OnItemTouchListener.this.mRecyclerView.getChildViewHolder(findChildViewUnder));
            return true;
        }
    }

    public OnItemTouchListener(RecyclerView recyclerView) {
        this.mRecyclerView = recyclerView;
        this.mGestureDetectorCompat = new GestureDetectorCompat(this.mRecyclerView.getContext(), new MyGestureListener());
    }

    @Override // androidx.recyclerview.widget.RecyclerView.OnItemTouchListener
    public boolean onInterceptTouchEvent(RecyclerView recyclerView, MotionEvent motionEvent) {
        this.mGestureDetectorCompat.onTouchEvent(motionEvent);
        return false;
    }

    public abstract void onItemClick(RecyclerView.ViewHolder viewHolder);

    @Override // androidx.recyclerview.widget.RecyclerView.OnItemTouchListener
    public void onRequestDisallowInterceptTouchEvent(boolean z) {
    }

    @Override // androidx.recyclerview.widget.RecyclerView.OnItemTouchListener
    public void onTouchEvent(RecyclerView recyclerView, MotionEvent motionEvent) {
        this.mGestureDetectorCompat.onTouchEvent(motionEvent);
    }
}
