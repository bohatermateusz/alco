package com.hismith.hismithapp.popup;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.popup.CommonPopupWindow;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.view.decoration.DividerItemDecoration;
import java.util.List;

/* loaded from: classes3.dex */
public class ActionAttachPopup {
    private ListAdapter mAdapter;
    private Context mContext;
    private FrameLayout mLayout;
    private LinearLayoutManager mLayoutManager;
    private List<ActionItem> mList;
    private RecyclerView mRecyclerView;
    private OnItemClickListener onItemClickListener;
    private CommonPopupWindow popupWindow;
    private View viewGroup;
    private int windowWidth;

    public static class ActionItem {
        Integer icon;
        String text;

        public ActionItem(Integer num, String str) {
            this.icon = num;
            this.text = str;
        }
    }

    class ListAdapter extends RecyclerView.Adapter<ViewHolder> {

        class ViewHolder extends RecyclerView.ViewHolder {
            private ImageView mIcon;
            private View mLine;
            private TextView mName;

            public ViewHolder(View view) {
                super(view);
                this.mIcon = (ImageView) view.findViewById(R.id.popup_img);
                this.mName = (TextView) view.findViewById(R.id.popup_txt);
                this.mLine = view.findViewById(R.id.popup_line);
            }
        }

        ListAdapter() {
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public int getItemCount() {
            return ActionAttachPopup.this.mList.size();
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
            onBindViewHolder2(viewHolder, i, (List<Object>) list);
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public void onBindViewHolder(ViewHolder viewHolder, final int i) {
            if (((ActionItem) ActionAttachPopup.this.mList.get(i)).icon != null) {
                viewHolder.mIcon.setImageResource(((ActionItem) ActionAttachPopup.this.mList.get(i)).icon.intValue());
                viewHolder.mIcon.setVisibility(0);
            } else {
                viewHolder.mIcon.setVisibility(8);
            }
            LanguageUtil.setText(ActionAttachPopup.this.mContext, viewHolder.mName, ((ActionItem) ActionAttachPopup.this.mList.get(i)).text);
            viewHolder.mLine.setVisibility(8);
            viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.popup.ActionAttachPopup.ListAdapter.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    if (ActionAttachPopup.this.popupWindow != null && ActionAttachPopup.this.popupWindow.isShowing()) {
                        ActionAttachPopup.this.popupWindow.dismiss();
                    }
                    if (ActionAttachPopup.this.onItemClickListener != null) {
                        ActionAttachPopup.this.onItemClickListener.onItemClick(view, i);
                    }
                }
            });
        }

        /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
        public void onBindViewHolder2(ViewHolder viewHolder, int i, List<Object> list) {
            if (list.isEmpty()) {
                onBindViewHolder(viewHolder, i);
            } else {
                viewHolder.mIcon.setVisibility(8);
                LanguageUtil.setText(ActionAttachPopup.this.mContext, viewHolder.mName, ((ActionItem) ActionAttachPopup.this.mList.get(i)).text);
            }
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
            return new ViewHolder(LayoutInflater.from(ActionAttachPopup.this.mContext).inflate(R.layout.item_play_mode_popup, viewGroup, false));
        }
    }

    public interface OnItemClickListener {
        void onItemClick(View view, int i);
    }

    public ActionAttachPopup(Context context, List<ActionItem> list) {
        this.mContext = context;
        this.mList = list;
        initView();
        initData();
        setListener();
    }

    private static int[] calculatePopWindowPos(View view, View view2) {
        int[] iArr = new int[3];
        int[] iArr2 = new int[2];
        view.getLocationOnScreen(iArr2);
        int height = view.getHeight();
        int screenHeight = MeasureUtil.getScreenHeight(view.getContext());
        int screenWidth = MeasureUtil.getScreenWidth(view.getContext());
        view2.measure(0, 0);
        int measuredHeight = view2.getMeasuredHeight();
        int measuredWidth = view2.getMeasuredWidth();
        int i = iArr2[1];
        if ((screenHeight - i) - height < measuredHeight) {
            iArr[0] = screenWidth - measuredWidth;
            iArr[1] = i - measuredHeight;
            iArr[2] = 1;
            return iArr;
        }
        iArr[0] = screenWidth - measuredWidth;
        iArr[1] = i + height;
        iArr[2] = 0;
        return iArr;
    }

    private void initData() {
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.mContext);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mRecyclerView.addItemDecoration(new DividerItemDecoration(this.mContext, 1));
        ListAdapter listAdapter = new ListAdapter();
        this.mAdapter = listAdapter;
        this.mRecyclerView.setAdapter(listAdapter);
    }

    private void initView() {
        View inflate = LayoutInflater.from(this.mContext).inflate(R.layout.dialog_action_attach_list, (ViewGroup) null, false);
        this.viewGroup = inflate;
        this.mLayout = (FrameLayout) inflate.findViewById(R.id.dialog_layout);
        this.mRecyclerView = (RecyclerView) this.viewGroup.findViewById(R.id.dialog_recycler_view);
    }

    private void setListener() {
    }

    public void dissPop() {
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow != null) {
            commonPopupWindow.dismiss();
        }
    }

    public void notifyDataSetChanged() {
        ListAdapter listAdapter = this.mAdapter;
        if (listAdapter != null) {
            listAdapter.notifyDataSetChanged();
        }
    }

    public void notifyItemChanged(int i) {
        ListAdapter listAdapter = this.mAdapter;
        if (listAdapter != null) {
            listAdapter.notifyItemChanged(i, "");
        }
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }

    public void showPop(View view, int i) {
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow == null || !commonPopupWindow.isShowing()) {
            this.popupWindow = new CommonPopupWindow.Builder(this.mContext).setView(this.viewGroup).setWidthAndHeight(-2, -2).setOutsideTouchable(true).create();
            int[] calculatePopWindowPos = calculatePopWindowPos(view, this.viewGroup);
            int i2 = calculatePopWindowPos[2];
            CommonPopupWindow commonPopupWindow2 = this.popupWindow;
            if (i2 == 1) {
                commonPopupWindow2.setAnimationStyle(R.style.AnimUp);
                this.mLayout.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_play_mode_action));
                calculatePopWindowPos[1] = calculatePopWindowPos[1] + MeasureUtil.dip2px(this.mContext, 5.0f);
            } else {
                commonPopupWindow2.setAnimationStyle(R.style.popmenu_animation);
                this.mLayout.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_play_mode_action));
                calculatePopWindowPos[1] = calculatePopWindowPos[1] - MeasureUtil.dip2px(this.mContext, 5.0f);
            }
            this.popupWindow.showAtLocation(view, 0, (calculatePopWindowPos[0] - MeasureUtil.dip2px(this.mContext, 13.0f)) + i, calculatePopWindowPos[1]);
        }
    }
}
