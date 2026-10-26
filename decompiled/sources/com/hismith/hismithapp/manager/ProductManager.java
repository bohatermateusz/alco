package com.hismith.hismithapp.manager;

import android.content.Context;
import android.text.TextUtils;
import com.google.gson.Gson;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.ProductsModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;

/* loaded from: classes3.dex */
public class ProductManager {
    public static ProductModel getProduct(String str) {
        str.hashCode();
        switch (str) {
            case "1001":
                return getProduct1001();
            case "1002":
                return getProduct1002();
            case "1003":
                return getProduct1003();
            case "1004":
                return getProduct1004();
            case "1005":
                return getProduct1005();
            case "1006":
                return getProduct1006();
            case "1007":
                return getProduct1007();
            case "1101":
                return getProduct1101();
            case "1102":
                return getProduct1102();
            case "1402":
                return getProduct1402();
            case "1501":
                return getProduct1501();
            case "2001":
                return getProduct2001();
            case "2101":
                return getProduct2101();
            case "2201":
                return getProduct2201();
            case "2202":
                return getProduct2202();
            case "2204":
                return getProduct2204();
            case "2205":
                return getProduct2205();
            case "3001":
                return getProduct3001();
            case "3101":
                return getProduct3101();
            case "3102":
                return getProduct3102();
            case "3302":
                return getProduct3302();
            case "4001":
                return getProduct4001();
            case "4002":
                return getProduct4002();
            case "7001":
                return getProduct7001();
            default:
                return null;
        }
    }

    private static ProductModel getProduct1001() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\":\"1\",\"code\":\"1001\",\"command\":\"AA\",\"name\":\"HISMITH\",\"shake\":\"50\",\"slide\":\"20\",\"climax\":\"0\",\"music\":\"0\",\"reversal\":\"0\",\"logo\":\"https:\\/\\/file.hismiths.com\\/product\\/HS062.png\",\"svgName\":\"01device\",\"svgVersion\":1711524867,\"svgUrl\":\"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/01device_left.svg\",\"svgLeftRight\":1,\"funs\":[{\"fCommand\":\"04\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\":\"SM\",\"modes\":[{\"mCommand\":\"AA050106\",\"locationIcon\":" + R.mipmap.mode1_1001_01 + "},{\"mCommand\":\"AA050207\",\"locationIcon\":" + R.mipmap.mode1_1001_02 + "},{\"mCommand\":\"AA050308\",\"locationIcon\":" + R.mipmap.mode1_1001_03 + "},{\"mCommand\":\"AA050409\",\"locationIcon\":" + R.mipmap.mode1_1001_04 + "},{\"mCommand\":\"AA05050A\",\"locationIcon\":" + R.mipmap.mode1_1001_05 + "},{\"mCommand\":\"AA05060B\",\"locationIcon\":" + R.mipmap.mode1_1001_06 + "},{\"mCommand\":\"AA05070C\",\"locationIcon\":" + R.mipmap.mode1_1001_07 + "},{\"mCommand\":\"AA05080D\",\"locationIcon\":" + R.mipmap.mode1_1001_08 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct1002() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\":\"2\",\"code\":\"1002\",\"command\":\"AA\",\"name\":\"Pro Traveler\",\"shake\":\"50\",\"slide\":\"20\",\"climax\":\"0\",\"music\":\"0\",\"reversal\":\"0\",\"logo\":\"https:\\/\\/file.hismiths.com\\/product\\/HS18.png\",\"svgName\":\"02device\",\"svgVersion\":1711524867,\"svgUrl\":\"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/02device.svg\",\"svgLeftRight\":0,\"funs\":[{\"fCommand\":\"04\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\":\"SM\",\"modes\":[{\"mCommand\":\"AA050106\",\"locationIcon\":" + R.mipmap.mode1_1002_01 + "},{\"mCommand\":\"AA050207\",\"locationIcon\":" + R.mipmap.mode1_1002_02 + "},{\"mCommand\":\"AA050308\",\"locationIcon\":" + R.mipmap.mode1_1002_03 + "},{\"mCommand\":\"AA050409\",\"locationIcon\":" + R.mipmap.mode1_1002_04 + "},{\"mCommand\":\"AA05050A\",\"locationIcon\":" + R.mipmap.mode1_1002_05 + "},{\"mCommand\":\"AA05060B\",\"locationIcon\":" + R.mipmap.mode1_1002_06 + "},{\"mCommand\":\"AA05070C\",\"locationIcon\":" + R.mipmap.mode1_1002_07 + "},{\"mCommand\":\"AA05080D\",\"locationIcon\":" + R.mipmap.mode1_1002_08 + "},{\"mCommand\":\"AA05090E\",\"locationIcon\":" + R.mipmap.mode1_1002_09 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct1003() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\":\"4\",\"code\":\"1003\",\"command\":\"AA\",\"name\":\"Sex Droid\",\"shake\":\"0\",\"slide\":\"20\",\"climax\":\"1\",\"music\":\"0\",\"reversal\":\"0\",\"logo\":\"https:\\/\\/file.hismiths.com\\/product\\/HS19.png\",\"svgName\":\"03device\",\"svgVersion\":1711524867,\"svgUrl\":\"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/03device.svg\",\"svgLeftRight\":0,\"funs\":[{\"fCommand\":\"04\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\":\"SM\",\"modes\":[{\"mCommand\":\"AA050106\",\"locationIcon\":" + R.mipmap.mode1_1003_01 + "},{\"mCommand\":\"AA050207\",\"locationIcon\":" + R.mipmap.mode1_1003_02 + "},{\"mCommand\":\"AA050308\",\"locationIcon\":" + R.mipmap.mode1_1003_03 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct1004() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"9\",\"code\": \"1004\",\"command\": \"CC\",\"name\": \"Hismith Mini\",\"shake\": \"50\",\"slide\": \"20\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/HS251.png\",\"svgName\": \"01device\",\"svgVersion\": 1711524867,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/01device_left.svg\",\"svgLeftRight\": 1,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_1004_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_1004_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_1004_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_1004_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_1004_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_1004_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_1004_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_1004_08 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct1005() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"10\",\"code\": \"1005\",\"command\": \"CC\",\"name\": \"Hismith S1\",\"shake\": \"50\",\"slide\": \"20\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/HS031.png\",\"svgName\": \"01device\",\"svgVersion\": 1711524867,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/01device_left.svg\",\"svgLeftRight\": 1,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_1005_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_1005_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_1005_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_1005_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_1005_05 + "},{\"mCommand\": \"CC04060a\",\"locationIcon\": " + R.mipmap.mode1_1005_06 + "},{\"mCommand\": \"CC04070b\",\"locationIcon\": " + R.mipmap.mode1_1005_07 + "},{\"mCommand\": \"CC04080c\",\"locationIcon\": " + R.mipmap.mode1_1005_08 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct1006() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"19\",\"code\": \"1006\",\"command\": \"AA\",\"name\": \"G011\",\"shake\": \"0\",\"slide\": \"50\",\"climax\": \"1\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/1006.png\",\"svgName\": \"06device\",\"svgVersion\": 1711524868,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/06device.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"04\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"AA050106\",\"locationIcon\": " + R.mipmap.mode1_1006_01 + "},{\"mCommand\": \"AA050207\",\"locationIcon\": " + R.mipmap.mode1_1006_02 + "},{\"mCommand\": \"AA050308\",\"locationIcon\": " + R.mipmap.mode1_1006_03 + "},{\"mCommand\": \"AA050409\",\"locationIcon\": " + R.mipmap.mode1_1006_04 + "},{\"mCommand\": \"AA05050A\",\"locationIcon\": " + R.mipmap.mode1_1006_05 + "}]},{\"fCommand\": \"\",\"fIconLocation\":" + R.mipmap.ficon_vibration + ",\"fType\": \"M\",\"modes\": [{\"mCommand\": \"AA060107\",\"locationIcon\": " + R.mipmap.mode2_1006_01 + "},{\"mCommand\": \"AA060208\",\"locationIcon\": " + R.mipmap.mode2_1006_02 + "},{\"mCommand\": \"AA060309\",\"locationIcon\": " + R.mipmap.mode2_1006_03 + "},{\"mCommand\": \"AA06040A\",\"locationIcon\": " + R.mipmap.mode2_1006_04 + "},{\"mCommand\": \"AA06050B\",\"locationIcon\": " + R.mipmap.mode2_1006_05 + "},{\"mCommand\": \"AA06060C\",\"locationIcon\": " + R.mipmap.mode2_1006_06 + "},{\"mCommand\": \"AA06070D\",\"locationIcon\": " + R.mipmap.mode2_1006_07 + "},{\"mCommand\": \"AA06080E\",\"locationIcon\": " + R.mipmap.mode2_1006_08 + "},{\"mCommand\": \"AA06090F\",\"locationIcon\": " + R.mipmap.mode2_1006_09 + "},{\"mCommand\": \"AA060A10\",\"locationIcon\": " + R.mipmap.mode2_1006_10 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct1007() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\":\"25\",\"code\":\"1007\",\"command\":\"CC\",\"name\":\"Premium 3.0 Max\",\"shake\":\"50\",\"slide\":\"20\",\"climax\":\"1\",\"music\":\"1\",\"reversal\":\"0\",\"logo\":\"https:\\/\\/file.hismiths.com\\/product\\/HS063.png\",\"svgName\":\"01device\",\"svgVersion\":1759201055,\"svgUrl\":\"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/01device_left.svg\",\"svgLeftRight\":1,\"funs\":[{\"fId\":\"35\",\"fCommand\":\"03\",\"fSpeedSet\":\"03\",\"fSpeedQuery\":\"03\",\"fMode\":\"04\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\":\"SM\",\"modes\":[{\"mCommand\":\"CC040105\",\"locationIcon\": " + R.mipmap.mode1_1007_01 + "},{\"mCommand\":\"CC040206\",\"locationIcon\": " + R.mipmap.mode1_1007_02 + "},{\"mCommand\":\"CC040307\",\"locationIcon\": " + R.mipmap.mode1_1007_03 + "},{\"mCommand\":\"CC040408\",\"locationIcon\": " + R.mipmap.mode1_1007_04 + "},{\"mCommand\":\"CC040509\",\"locationIcon\": " + R.mipmap.mode1_1007_05 + "},{\"mCommand\":\"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_1007_06 + "},{\"mCommand\":\"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_1007_07 + "},{\"mCommand\":\"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_1007_08 + "}]}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct1101() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"7\",\"code\": \"1101\",\"command\": \"CC\",\"name\": \"Hismith S2\",\"shake\": \"0\",\"slide\": \"20\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/HS052.png\",\"svgName\": \"01device\",\"svgVersion\": 1711524867,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/01device_left.svg\",\"svgLeftRight\": 1,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SMS\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_1101_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_1101_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_1101_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_1101_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_1101_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_1101_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_1101_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_1101_08 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct1102() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"22\",\"code\": \"1102\",\"command\": \"CC\",\"name\": \"HISMITH S3\",\"shake\": \"50\",\"slide\": \"50\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/55BE0596BBED216D048FCE18326E.jpg\",\"svgName\": \"01device\",\"svgVersion\": 1711524867,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/01device_left.svg\",\"svgLeftRight\": 1,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_1102_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_1102_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_1102_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_1102_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_1102_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_1102_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_1102_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_1102_08 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct1402() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"20\",\"code\": \"1402\",\"command\": \"CC\",\"name\": \"UKULELE\",\"shake\": \"50\",\"slide\": \"20\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/AX02APP.png\",\"svgName\": \"01device\",\"svgVersion\": 1711524867,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/01device_left.svg\",\"svgLeftRight\": 1,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_1402_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_1402_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_1402_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_1402_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_1402_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_1402_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_1402_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_1402_08 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct1501() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"21\",\"code\": \"1501\",\"command\": \"CC\",\"name\": \"PleasureDrive\",\"shake\": \"50\",\"slide\": \"20\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/HS36.png\",\"svgName\": \"01device\",\"svgVersion\": 1711524867,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/01device_left.svg\",\"svgLeftRight\": 1,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_1501_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_1501_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_1501_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_1501_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_1501_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_1501_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_1501_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_1501_08 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct2001() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\":\"5\",\"code\":\"2001\",\"command\":\"AA\",\"name\":\"AIM Cup\",\"shake\":\"100\",\"slide\":\"100\",\"climax\":\"0\",\"music\":\"1\",\"reversal\":\"0\",\"logo\":\"https:\\/\\/file.hismiths.com\\/product\\/HSA1011.png\",\"svgName\":\"06device\",\"svgVersion\":1711524868,\"svgUrl\":\"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/06device.svg\",\"svgLeftRight\":0,\"funs\":[{\"fCommand\":\"04\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\":\"SM\",\"modes\":[{\"mCommand\":\"AA050106\",\"locationIcon\":" + R.mipmap.mode1_2001_01 + "},{\"mCommand\":\"AA050207\",\"locationIcon\":" + R.mipmap.mode1_2001_02 + "},{\"mCommand\":\"AA050308\",\"locationIcon\":" + R.mipmap.mode1_2001_03 + "},{\"mCommand\":\"AA050409\",\"locationIcon\":" + R.mipmap.mode1_2001_04 + "},{\"mCommand\":\"AA05050A\",\"locationIcon\":" + R.mipmap.mode1_2001_05 + "},{\"mCommand\":\"AA05060B\",\"locationIcon\":" + R.mipmap.mode1_2001_06 + "},{\"mCommand\":\"AA05070C\",\"locationIcon\":" + R.mipmap.mode1_2001_07 + "},{\"mCommand\":\"AA05080D\",\"locationIcon\":" + R.mipmap.mode1_2001_08 + "},{\"mCommand\":\"AA05090E\",\"locationIcon\":" + R.mipmap.mode1_2001_09 + "},{\"mCommand\":\"AA050A0F\",\"locationIcon\":" + R.mipmap.mode1_2001_10 + "}]},{\"fCommand\":\"\",\"fIconLocation\":" + R.mipmap.ficon_vibration + ",\"fType\":\"M\",\"modes\":[{\"mCommand\":\"AA060107\",\"locationIcon\":" + R.mipmap.mode2_2001_01 + "},{\"mCommand\":\"AA060208\",\"locationIcon\":" + R.mipmap.mode2_2001_02 + "},{\"mCommand\":\"AA060309\",\"locationIcon\":" + R.mipmap.mode2_2001_03 + "},{\"mCommand\":\"AA06040A\",\"locationIcon\":" + R.mipmap.mode2_2001_04 + "},{\"mCommand\":\"AA06050B\",\"locationIcon\":" + R.mipmap.mode2_2001_05 + "},{\"mCommand\":\"AA06060C\",\"locationIcon\":" + R.mipmap.mode2_2001_06 + "},{\"mCommand\":\"AA06070D\",\"locationIcon\":" + R.mipmap.mode2_2001_07 + "},{\"mCommand\":\"AA06080E\",\"locationIcon\":" + R.mipmap.mode2_2001_08 + "},{\"mCommand\":\"AA06090F\",\"locationIcon\":" + R.mipmap.mode2_2001_09 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct2101() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"12\",\"code\": \"2101\",\"command\": \"CC\",\"name\": \"Eropair Cup\",\"shake\": \"100\",\"slide\": \"100\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/EROP0012.png\",\"svgName\": \"09device\",\"svgVersion\": 1711524868,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/09device.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_sucking + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_2101_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_2101_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_2101_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_2101_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_2101_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_2101_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_2101_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_2101_08 + "},{\"mCommand\": \"CC04090D\",\"locationIcon\": " + R.mipmap.mode1_2101_09 + "},{\"mCommand\": \"CC040A0E\",\"locationIcon\": " + R.mipmap.mode1_2101_10 + "}]},{\"fCommand\": \"05\",\"fIconLocation\":" + R.mipmap.ficon_vibration + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC060107\",\"locationIcon\": " + R.mipmap.mode2_2101_01 + "},{\"mCommand\": \"CC060208\",\"locationIcon\": " + R.mipmap.mode2_2101_02 + "},{\"mCommand\": \"CC060309\",\"locationIcon\": " + R.mipmap.mode2_2101_03 + "},{\"mCommand\": \"CC06040A\",\"locationIcon\": " + R.mipmap.mode2_2101_04 + "},{\"mCommand\": \"CC06050B\",\"locationIcon\": " + R.mipmap.mode2_2101_05 + "},{\"mCommand\": \"CC06060C\",\"locationIcon\": " + R.mipmap.mode2_2101_06 + "},{\"mCommand\": \"CC06070D\",\"locationIcon\": " + R.mipmap.mode2_2101_07 + "},{\"mCommand\": \"CC06080E\",\"locationIcon\": " + R.mipmap.mode2_2101_08 + "},{\"mCommand\": \"CC06090F\",\"locationIcon\": " + R.mipmap.mode2_2101_09 + "},{\"mCommand\": \"CC060A10\",\"locationIcon\": " + R.mipmap.mode2_2101_10 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct2201() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"8\",\"code\": \"2201\",\"command\": \"CC\",\"name\": \"Sinloli\",\"shake\": \"100\",\"slide\": \"100\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/SA0011.png\",\"svgName\": \"08device\",\"svgVersion\": 1711524868,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/08device.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_sucking + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_2201_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_2201_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_2201_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_2201_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_2201_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_2201_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_2201_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_2201_08 + "}]},{\"fCommand\": \"05\",\"fIconLocation\":" + R.mipmap.ficon_vibration + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC060107\",\"locationIcon\": " + R.mipmap.mode2_2201_01 + "},{\"mCommand\": \"CC060208\",\"locationIcon\": " + R.mipmap.mode2_2201_02 + "},{\"mCommand\": \"CC060309\",\"locationIcon\": " + R.mipmap.mode2_2201_03 + "},{\"mCommand\": \"CC06040A\",\"locationIcon\": " + R.mipmap.mode2_2201_04 + "},{\"mCommand\": \"CC06050B\",\"locationIcon\": " + R.mipmap.mode2_2201_05 + "},{\"mCommand\": \"CC06060C\",\"locationIcon\": " + R.mipmap.mode2_2201_06 + "},{\"mCommand\": \"CC06070D\",\"locationIcon\": " + R.mipmap.mode2_2201_07 + "},{\"mCommand\": \"CC06080E\",\"locationIcon\": " + R.mipmap.mode2_2201_08 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct2202() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"14\",\"code\": \"2202\",\"command\": \"CC\",\"name\": \"Sinloli\",\"shake\": \"100\",\"slide\": \"100\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/SA0041.png\",\"svgName\": \"08device\",\"svgVersion\": 1711524868,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/08device.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_2202_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_2202_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_2202_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_2202_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_2202_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_2202_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_2202_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_2202_08 + "},{\"mCommand\": \"CC04090D\",\"locationIcon\": " + R.mipmap.mode1_2202_09 + "},{\"mCommand\": \"CC040A0E\",\"locationIcon\": " + R.mipmap.mode1_2202_10 + "}]},{\"fCommand\": \"05\",\"fIconLocation\":" + R.mipmap.ficon_vibration + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC060107\",\"locationIcon\": " + R.mipmap.mode2_2202_01 + "},{\"mCommand\": \"CC060208\",\"locationIcon\": " + R.mipmap.mode2_2202_02 + "},{\"mCommand\": \"CC060309\",\"locationIcon\": " + R.mipmap.mode2_2202_03 + "},{\"mCommand\": \"CC06040A\",\"locationIcon\": " + R.mipmap.mode2_2202_04 + "},{\"mCommand\": \"CC06050B\",\"locationIcon\": " + R.mipmap.mode2_2202_05 + "},{\"mCommand\": \"CC06060C\",\"locationIcon\": " + R.mipmap.mode2_2202_06 + "},{\"mCommand\": \"CC06070D\",\"locationIcon\": " + R.mipmap.mode2_2202_07 + "},{\"mCommand\": \"CC06080E\",\"locationIcon\": " + R.mipmap.mode2_2202_08 + "},{\"mCommand\": \"CC06090F\",\"locationIcon\": " + R.mipmap.mode2_2202_09 + "},{\"mCommand\": \"CC060A10\",\"locationIcon\": " + R.mipmap.mode2_2202_10 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct2204() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"16\",\"code\": \"2204\",\"command\": \"CC\",\"name\": \"Sinloli Cosima\",\"shake\": \"100\",\"slide\": \"100\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/SA006.png\",\"svgName\": \"09device\",\"svgVersion\": 1711524868,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/09device.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_sucking + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_2204_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_2204_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_2204_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_2204_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_2204_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_2204_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_2204_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_2204_08 + "},{\"mCommand\": \"CC04090D\",\"locationIcon\": " + R.mipmap.mode1_2204_09 + "},{\"mCommand\": \"CC040A0E\",\"locationIcon\": " + R.mipmap.mode1_2204_10 + "}]},{\"fCommand\": \"05\",\"fIconLocation\":" + R.mipmap.ficon_vibration + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC060107\",\"locationIcon\": " + R.mipmap.mode2_2204_01 + "},{\"mCommand\": \"CC060208\",\"locationIcon\": " + R.mipmap.mode2_2204_02 + "},{\"mCommand\": \"CC060309\",\"locationIcon\": " + R.mipmap.mode2_2204_03 + "},{\"mCommand\": \"CC06040A\",\"locationIcon\": " + R.mipmap.mode2_2204_04 + "},{\"mCommand\": \"CC06050B\",\"locationIcon\": " + R.mipmap.mode2_2204_05 + "},{\"mCommand\": \"CC06060C\",\"locationIcon\": " + R.mipmap.mode2_2204_06 + "},{\"mCommand\": \"CC06070D\",\"locationIcon\": " + R.mipmap.mode2_2204_07 + "},{\"mCommand\": \"CC06080E\",\"locationIcon\": " + R.mipmap.mode2_2204_08 + "},{\"mCommand\": \"CC06090F\",\"locationIcon\": " + R.mipmap.mode2_2204_09 + "},{\"mCommand\": \"CC060A10\",\"locationIcon\": " + R.mipmap.mode2_2204_10 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct2205() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"17\",\"code\": \"2205\",\"command\": \"CC\",\"name\": \"Sinloli Aston\",\"shake\": \"100\",\"slide\": \"100\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"1\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/SA0072.png\",\"svgName\": \"06device\",\"svgVersion\": 1711524868,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/06device.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_2205_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_2205_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_2205_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_2205_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_2205_05 + "},{\"mCommand\": \"CC04060a\",\"locationIcon\": " + R.mipmap.mode1_2205_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_2205_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_2205_08 + "}]},{\"fCommand\": \"05\",\"fIconLocation\":" + R.mipmap.ficon_rotate + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC060107\",\"locationIcon\": " + R.mipmap.mode2_2205_01 + "},{\"mCommand\": \"CC060208\",\"locationIcon\": " + R.mipmap.mode2_2205_02 + "},{\"mCommand\": \"CC060309\",\"locationIcon\": " + R.mipmap.mode2_2205_03 + "},{\"mCommand\": \"CC06040A\",\"locationIcon\": " + R.mipmap.mode2_2205_04 + "},{\"mCommand\": \"CC06050B\",\"locationIcon\": " + R.mipmap.mode2_2205_05 + "},{\"mCommand\": \"CC06060C\",\"locationIcon\": " + R.mipmap.mode2_2205_06 + "},{\"mCommand\": \"CC06070D\",\"locationIcon\": " + R.mipmap.mode2_2205_07 + "},{\"mCommand\": \"CC06080E\",\"locationIcon\": " + R.mipmap.mode2_2205_08 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct3001() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"6\",\"code\": \"3001\",\"command\": \"AA\",\"name\": \"Wildolo\",\"shake\": \"100\",\"slide\": \"100\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/WDA006LF6.png\",\"svgName\": \"05device\",\"svgVersion\": 1711524868,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/05device.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"04\",\"fIconLocation\":" + R.mipmap.ficon_vibration + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"AA050106\",\"locationIcon\": " + R.mipmap.mode1_3001_01 + "},{\"mCommand\": \"AA050207\",\"locationIcon\": " + R.mipmap.mode1_3001_02 + "},{\"mCommand\": \"AA050308\",\"locationIcon\": " + R.mipmap.mode1_3001_03 + "},{\"mCommand\": \"AA050409\",\"locationIcon\": " + R.mipmap.mode1_3001_04 + "},{\"mCommand\": \"AA05050A\",\"locationIcon\": " + R.mipmap.mode1_3001_05 + "},{\"mCommand\": \"AA05060B\",\"locationIcon\": " + R.mipmap.mode1_3001_06 + "},{\"mCommand\": \"AA05070C\",\"locationIcon\": " + R.mipmap.mode1_3001_07 + "},{\"mCommand\": \"AA05080D\",\"locationIcon\": " + R.mipmap.mode1_3001_08 + "},{\"mCommand\": \"AA05090E\",\"locationIcon\": " + R.mipmap.mode1_3001_09 + "},{\"mCommand\": \"AA050A0F\",\"locationIcon\": " + R.mipmap.mode1_3001_10 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct3101() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"11\",\"code\": \"3101\",\"command\": \"CC\",\"name\": \"Eropair V1\",\"shake\": \"100\",\"slide\": \"100\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/EROP0031.png\",\"svgName\": \"04device\",\"svgVersion\": 1711524867,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/04device.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_bigvibration + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_3101_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_3101_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_3101_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_3101_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_3101_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_3101_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_3101_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_3101_08 + "},{\"mCommand\": \"CC04090D\",\"locationIcon\": " + R.mipmap.mode1_3101_09 + "},{\"mCommand\": \"CC040A0E\",\"locationIcon\": " + R.mipmap.mode1_3101_10 + "}]},{\"fCommand\": \"05\",\"fIconLocation\":" + R.mipmap.ficon_smallvibration + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC060107\",\"locationIcon\": " + R.mipmap.mode2_3101_01 + "},{\"mCommand\": \"CC060208\",\"locationIcon\": " + R.mipmap.mode2_3101_02 + "},{\"mCommand\": \"CC060309\",\"locationIcon\": " + R.mipmap.mode2_3101_03 + "},{\"mCommand\": \"CC06040A\",\"locationIcon\": " + R.mipmap.mode2_3101_04 + "},{\"mCommand\": \"CC06050B\",\"locationIcon\": " + R.mipmap.mode2_3101_05 + "},{\"mCommand\": \"CC06060C\",\"locationIcon\": " + R.mipmap.mode2_3101_06 + "},{\"mCommand\": \"CC06070D\",\"locationIcon\": " + R.mipmap.mode2_3101_07 + "},{\"mCommand\": \"CC06080E\",\"locationIcon\": " + R.mipmap.mode2_3101_08 + "},{\"mCommand\": \"CC06090F\",\"locationIcon\": " + R.mipmap.mode2_3101_09 + "},{\"mCommand\": \"CC060A10\",\"locationIcon\": " + R.mipmap.mode2_3101_10 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct3102() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"15\",\"code\": \"3102\",\"command\": \"CC\",\"name\": \"Eropair S1\",\"shake\": \"100\",\"slide\": \"100\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/EROP0021.png\",\"svgName\": \"07device\",\"svgVersion\": 1711524868,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/07device.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_3102_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_3102_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_3102_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_3102_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_3102_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_3102_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_3102_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_3102_08 + "},{\"mCommand\": \"CC04090D\",\"locationIcon\": " + R.mipmap.mode1_3102_09 + "},{\"mCommand\": \"CC040A0E\",\"locationIcon\": " + R.mipmap.mode1_3102_10 + "}]},{\"fCommand\": \"05\",\"fIconLocation\":" + R.mipmap.ficon_vibration + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC060107\",\"locationIcon\": " + R.mipmap.mode2_3102_01 + "},{\"mCommand\": \"CC060208\",\"locationIcon\": " + R.mipmap.mode2_3102_02 + "},{\"mCommand\": \"CC060309\",\"locationIcon\": " + R.mipmap.mode2_3102_03 + "},{\"mCommand\": \"CC06040A\",\"locationIcon\": " + R.mipmap.mode2_3102_04 + "},{\"mCommand\": \"CC06050B\",\"locationIcon\": " + R.mipmap.mode2_3102_05 + "},{\"mCommand\": \"CC06060C\",\"locationIcon\": " + R.mipmap.mode2_3102_06 + "},{\"mCommand\": \"CC06070D\",\"locationIcon\": " + R.mipmap.mode2_3102_07 + "},{\"mCommand\": \"CC06080E\",\"locationIcon\": " + R.mipmap.mode2_3102_08 + "},{\"mCommand\": \"CC06090F\",\"locationIcon\": " + R.mipmap.mode2_3102_09 + "},{\"mCommand\": \"CC060A10\",\"locationIcon\": " + R.mipmap.mode2_3102_10 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct3302() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"23\",\"code\": \"3302\",\"command\": \"CC\",\"name\": \"Wildolo-Pro\",\"shake\": \"100\",\"slide\": \"100\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/icon2.png\",\"svgName\": \"05device\",\"svgVersion\": 1711524868,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/05device.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_vibration + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_3302_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_3302_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_3302_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_3302_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_3302_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_3302_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_3302_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_3302_08 + "},{\"mCommand\": \"CC04090D\",\"locationIcon\": " + R.mipmap.mode1_3302_09 + "},{\"mCommand\": \"CC040A0E\",\"locationIcon\": " + R.mipmap.mode1_3302_10 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct4001() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"13\",\"code\": \"4001\",\"command\": \"CC\",\"name\": \"Auxfun Box\",\"shake\": \"50\",\"slide\": \"20\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/STOY04302.png\",\"svgName\": \"01device\",\"svgVersion\": 1711524867,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/01device_left.svg\",\"svgLeftRight\": 1,\"funs\": [{\"fCommand\": \"03\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\": \"SM\",\"modes\": [{\"mCommand\": \"CC040105\",\"locationIcon\": " + R.mipmap.mode1_4001_01 + "},{\"mCommand\": \"CC040206\",\"locationIcon\": " + R.mipmap.mode1_4001_02 + "},{\"mCommand\": \"CC040307\",\"locationIcon\": " + R.mipmap.mode1_4001_03 + "},{\"mCommand\": \"CC040408\",\"locationIcon\": " + R.mipmap.mode1_4001_04 + "},{\"mCommand\": \"CC040509\",\"locationIcon\": " + R.mipmap.mode1_4001_05 + "},{\"mCommand\": \"CC04060A\",\"locationIcon\": " + R.mipmap.mode1_4001_06 + "},{\"mCommand\": \"CC04070B\",\"locationIcon\": " + R.mipmap.mode1_4001_07 + "},{\"mCommand\": \"CC04080C\",\"locationIcon\": " + R.mipmap.mode1_4001_08 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct4002() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\": \"18\",\"code\": \"4002\",\"command\": \"CC\",\"name\": \"Hismith PiuPiu\",\"shake\": \"0\",\"slide\": \"0\",\"climax\": \"0\",\"music\": \"0\",\"reversal\": \"0\",\"logo\": \"https:\\/\\/file.hismiths.com\\/product\\/piupiu.png\",\"svgName\": \"PiuPiu\",\"svgVersion\": 1730778891,\"svgUrl\": \"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/PiuPiu.svg\",\"svgLeftRight\": 0,\"funs\": [{\"fCommand\": \"0B\",\"fIconLocation\":" + R.mipmap.ficon_piupiu + ",\"fType\": \"F\",\"modes\": [{\"mCommand\": \"CC0B010C\"}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static ProductModel getProduct7001() {
        try {
            return (ProductModel) new Gson().fromJson("{\"id\":\"24\",\"code\":\"7001\",\"command\":\"AA\",\"name\":\"G081\",\"shake\":\"100\",\"slide\":\"100\",\"climax\":\"0\",\"music\":\"0\",\"reversal\":\"0\",\"logo\":\"https:\\/\\/file.hismiths.com\\/product\\/none.gif\",\"svgName\":\"05device\",\"svgVersion\":1711524868,\"svgUrl\":\"https:\\/\\/file.hismiths.com\\/apm\\/svg\\/app\\/05device.svg\",\"svgLeftRight\":0,\"funs\":[{\"fId\":\"33\",\"fCommand\":\"04\",\"fSpeedSet\":\"04\",\"fSpeedQuery\":\"03\",\"fMode\":\"05\",\"fIconLocation\":" + R.mipmap.ficon_impact + ",\"fType\":\"SM\",\"modes\":[{\"mCommand\":\"AA050106\",\"locationIcon\": " + R.mipmap.mode1_7001_01 + "},{\"mCommand\":\"AA050207\",\"locationIcon\": " + R.mipmap.mode1_7001_02 + "},{\"mCommand\":\"AA050308\",\"locationIcon\": " + R.mipmap.mode1_7001_03 + "},{\"mCommand\":\"AA050409\",\"locationIcon\": " + R.mipmap.mode1_7001_04 + "},{\"mCommand\":\"AA05050A\",\"locationIcon\": " + R.mipmap.mode1_7001_05 + "},{\"mCommand\":\"AA05060B\",\"locationIcon\": " + R.mipmap.mode1_7001_06 + "},{\"mCommand\":\"AA05070C\",\"locationIcon\": " + R.mipmap.mode1_7001_07 + "}]}]}", ProductModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static ProductModel getProduct_v1(Context context, String str) {
        if (TextUtils.isEmpty(str)) {
            return new ProductModel();
        }
        ProductsModel productsModel = (ProductsModel) new Gson().fromJson((String) SharedPreferencesUtil.get(context, Constans.ProductAllData, ""), ProductsModel.class);
        if (productsModel != null && productsModel.getProducts() != null && productsModel.getProducts().length > 0) {
            for (int i = 0; i < productsModel.getProducts().length; i++) {
                if (productsModel.getProducts()[i].getCode().equals(str)) {
                    return productsModel.getProducts()[i];
                }
            }
        }
        return getProduct(str);
    }
}
