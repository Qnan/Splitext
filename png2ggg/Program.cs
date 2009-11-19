using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.IO;

namespace png2ggg
{
   class Program
   {
      static void Main(string[] args)
      {
         Bitmap img = Image.FromFile(args[0]) as Bitmap;
         FileStream fs = new FileStream(args[1], FileMode.Create);
         BinaryWriter bw = new BinaryWriter(fs);
         bw.Write(img.Width);
         bw.Write(img.Height);
         bw.Write((int)img.HorizontalResolution);
         bw.Write((int)img.VerticalResolution);

         for (int y = 0; y < img.Height; ++y)
            for (int x = 0; x < img.Width; ++x)
               bw.Write((int)Math.Min(Math.Max(img.GetPixel(x, y).GetBrightness()*255,0),255));
         bw.Close();
      }                 
   }
}
