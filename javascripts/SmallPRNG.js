/**
  SmallPRNG library for JavaScript. Not special,
  not optimized, not perfect. It produces pseudo-
  random numbers by manipulating a context which
  was initialized with a seed.

  MIT licensed, use wherever you like, but a link
  back to my CodePen (ImagineProgramming) or website
  is appreciated (http://www.imagine-programming.com)

  Based on:
  http://burtleburtle.net/bob/rand/smallprng.html
*/

(function(target, name) {
    // maximum randval for .random([max/min [, max]])
    var RMAX = 0x7FFFFFFF;

    var floor = Math.floor,
        abs = Math.abs;

    // 32-bits rotate
    var rot = function(x, k) {
      return (((x << k) & 0xffffffff) | (x >>> (32 - k)))>>>0;
    };

    // constructor for the context
    var SmallPRNG = function(seed) {
      this.a = 0xf1ea5eed;
      this.b = seed;
      this.c = seed;
      this.d = seed;
      this.s = 0;
    };

    // reseed the context
    SmallPRNG.prototype.seed = function(seed) {
      this.a = 0xf1ea5eed;
      this.b = seed;
      this.c = seed;
      this.d = seed;
      this.s = 0;
    };

    // reseed the context b, c and d fields
    SmallPRNG.prototype.seedAll = function(b, c, d) {
      this.a = 0xf1ea5eed;
      this.b = b;
      this.c = c;
      this.d = d;
      this.s = 0;
    };

    // get the next randval in the ctx
    SmallPRNG.prototype.randval = function() {
      // notice the >>>0, which is to get an unsigned integer.
      var e  = ((this.a - rot(this.b, 27)) & 0xffffffff)>>>0;
      this.a = ((this.b ^ rot(this.c, 17)) & 0xffffffff)>>>0;
      this.b = ((this.c + this.d) & 0xffffffff)>>>0;
      this.c = ((this.d + e) & 0xffffffff)>>>0;
      this.d = ((e + this.a) & 0xffffffff)>>>0;
      this.s++;
      return this.d;
    };

    // step `times' times in the CTX
    SmallPRNG.prototype.step = function(times) {
      times = (typeof(times) === "number" ? times : 1);
      if(times === 0) {
        times = 1;
      }

      for(var i = 0; i < times; i++) {
        var e  = ((this.a - rot(this.b, 27)) & 0xffffffff)>>>0;
        this.a = ((this.b ^ rot(this.c, 17)) & 0xffffffff)>>>0;
        this.b = ((this.c + this.d) & 0xffffffff)>>>0;
        this.c = ((this.d + e) & 0xffffffff)>>>0;
        this.d = ((e + this.a) & 0xffffffff)>>>0;
        this.s++;
      }
    };

    SmallPRNG.prototype.random = function() {
      var r = ((this.randval() % RMAX) / RMAX);
      switch(arguments.length) {
        // zero arguments, return the 0-1 random factor
        case 0: {
          return r;
        } break;

        // 1 argument (max val), return random between 1 and max
        case 1: {
          var u = arguments[0];
          if(u < 1) {
            console.log("upper limit invalid");
            return null;
          }

          return (floor(r * u) + 1);

        } break;

        // 2 arguments (min, max val), return random between min and max
        case 2: {
          var l = arguments[0];
          var u = arguments[1];

          if(l >= u) {
            console.log("upper limit invalid");
            return null;
          }

          return (floor(r * (u - l + 1)) + l);
        } break;

        default: {
          console.log("invalid amount of arguments");
        } break;
      }

      return null;
    };

    target[name] = SmallPRNG;
  }(window, "SmallPRNG"));
