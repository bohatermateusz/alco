package com.hismith.hismithapp.manager;

import android.content.Context;
import android.graphics.Bitmap;
import com.hismith.hismithapp.R;
import com.tencent.mm.opensdk.modelmsg.SendMessageToWX;
import com.tencent.mm.opensdk.modelmsg.WXMediaMessage;
import com.tencent.mm.opensdk.modelmsg.WXTextObject;
import com.tencent.mm.opensdk.modelmsg.WXWebpageObject;
import com.tencent.mm.opensdk.openapi.IWXAPI;
import com.tencent.mm.opensdk.openapi.WXAPIFactory;

/* loaded from: classes3.dex */
public class WxShareUtils {
    public static boolean isWXAppInstalled(Context context) {
        return WXAPIFactory.createWXAPI(context, context.getResources().getString(R.string.wx_id)).isWXAppInstalled();
    }

    public static void shareText(Context context, String str) {
        IWXAPI createWXAPI = WXAPIFactory.createWXAPI(context, context.getResources().getString(R.string.wx_id));
        if (createWXAPI.isWXAppInstalled()) {
            WXTextObject wXTextObject = new WXTextObject();
            wXTextObject.text = str;
            WXMediaMessage wXMediaMessage = new WXMediaMessage();
            wXMediaMessage.mediaObject = wXTextObject;
            wXMediaMessage.description = str;
            SendMessageToWX.Req req = new SendMessageToWX.Req();
            req.transaction = "text";
            req.message = wXMediaMessage;
            req.scene = 0;
            createWXAPI.sendReq(req);
        }
    }

    public static void shareWeb(Context context, String str, String str2, String str3, Bitmap bitmap) {
        IWXAPI createWXAPI = WXAPIFactory.createWXAPI(context, context.getResources().getString(R.string.wx_id));
        if (createWXAPI.isWXAppInstalled()) {
            WXWebpageObject wXWebpageObject = new WXWebpageObject();
            wXWebpageObject.webpageUrl = str;
            WXMediaMessage wXMediaMessage = new WXMediaMessage(wXWebpageObject);
            wXMediaMessage.title = str2;
            wXMediaMessage.description = str3;
            wXMediaMessage.setThumbImage(bitmap);
            SendMessageToWX.Req req = new SendMessageToWX.Req();
            req.transaction = "webpage";
            req.message = wXMediaMessage;
            req.scene = 0;
            createWXAPI.sendReq(req);
        }
    }
}
