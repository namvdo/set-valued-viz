from _imports import *
SAVE_DIR = os.path.join(WORKDIR, "mask_model_saves")

class ModelStatistics():
    escaped_points = 0
    escaped_points_total = 0
    processed_points = 0
    processed_points_total = 0
    hausdorff_distance = 0
    
    def __init__(self):
        self.timeline = [] # ? log various events & configuration changes at timesteps

    def log_processed_points(self, n):
        self.processed_points = n
        self.processed_points_total += n
        
    def log_escaped_points(self, n):
        self.escaped_points = n
        self.escaped_points_total += n

    def log_hausdorff(self, dist):
        self.hausdorff_distance = dist

class ModelConfiguration():
    alt_visuals = False
    autofill = False
    
    precision = 0.005
    start_point = UserModifiablePoint(0,0)
    escape_distance = 10.
    # distance at which to consider them as escaped (prevents extreme array sizes)
    
    border_width = 1
    epsilon_radius = 0.1
    timestep = 0
    
    image_history = None
    image = None
    
    topleft = (0,0)
    bottomright = (0,0)
    topleft_old = (0,0)

    def __init__(self):
        self.padding_error = np.zeros(2)
        
        # place the starting point on the mask
        self.image = circle_mask(0, border=self.border_width)
        
        tl = -np.divide(self.image.shape, 2).astype(np.int32).astype(np.float64)
        br = np.add(self.image.shape, tl)
        self.topleft = tl*self.precision
        self.bottomright = br*self.precision
        self.topleft = np.add(self.topleft, (self.start_point.x,-self.start_point.y))
        self.bottomright = np.add(self.bottomright, (self.start_point.x,-self.start_point.y))
        #
        
        self.image_history = self.image.astype(np.uint16)
        self.topleft_old = self.topleft.copy()
        self.timestep = 0

    def epsilon_circle(self):
        return circle_mask(int(self.epsilon_radius/self.precision), border=self.border_width)

    def calc_current_border_points(self):
        border = get_mask_border(self.image)
        indexes = calc_mask_indexes(self.image)
        return indexes[border].astype(np.float64)

    def points_as_units(self, points):
        points -= self.topleft
        points /= self.precision

    def points_as_values(self, points):
        points *= self.precision
        points += self.topleft
    
    def visualization(self):
        axis_range = (self.topleft[0],self.bottomright[0],-self.bottomright[1],-self.topleft[1])
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.image_history.swapaxes(0, 1), extent=axis_range)
        ax[1].imshow(self.image.swapaxes(0, 1), extent=axis_range)
        ax[0].set_title(f"Cumulative Timesteps")
        ax[1].set_title(f"Timestep: {self.timestep}")
        plt.show()

class ConsoleInterface():
    interface_depth = 0
    autorun_timesteps = 0
    autorun_hausdorff = 0
    
    def __init__(self):
        self.stats = ModelStatistics()
        self.config = ModelConfiguration()
        self.function = UserModifiableMappingFunction()

    def _show_user_options(self, desc, options):
        desc = clean_text_box(desc)
        
        string = ""
        for i,x in enumerate(options):
            if i!=0: string += "\n"
            string += str(i)+". "+x[0]
        
        string = clean_text_box(string, l="* ", pad_t=0, pad_b=0)
        self._depth_print(desc+"\n"+string)
        
    def _depth_print(self, string, offset=0):
        depth = " "*(max(self.interface_depth+offset, 0)*4)
        string = string.replace("\n", "\n"+depth)
        print(depth+string)
        
    def _depth_input(self, string, *args, offset=0, **kwargs):
        depth = " "*(max(self.interface_depth+offset, 0)*4)
        string = string.replace("\n", "\n"+depth)
        return input(depth+string, *args, **kwargs)

    def _process_user_input(self, options):
        user_input = self._depth_input("SELECT: ")
        if user_input and user_input.isnumeric():
            user_input = int(user_input)
            if 0<=user_input<len(options):
                f = options[user_input][1]
                if f is not None:
                    self._depth_print("-> "+options[user_input][0])
                    self.interface_depth += 1
                    output = f()
                    self.interface_depth -= 1
                    return output
                else:
                    self._depth_print("Not implemented")
            else:
                self._depth_print("Not a valid option")
        else: self._depth_print("Invalid input")
        print("")
        return False

    def quit(self): return True

    def mainmenu(self):
        while not self.mainmenu_options(): pass
    
    def mainmenu_options(self):
        options = [
            ("Start", self.activerun),
            ("Set Precision", self.set_precision),
            ("Set Start Point", self.set_startpoint),
            ("Set Escape Distance", self.set_escapedistance),
##            ("Load", None),
            ("Exit", self.quit),
            ]
        desc = "Simple interface for dynamic model visualization"
        desc += f"\n\nPrecision: {self.config.precision}"
        desc += f"\nStarting point: {self.config.start_point}"
        desc += f"\nEscape distance: {self.config.escape_distance} ({int(self.config.escape_distance/self.config.precision)} pixels)"
        self._show_user_options(desc, options)
        return self._process_user_input(options)




    
    def set_precision(self):
        desc = "Precision of the simulation"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("precision", target=self.config)
        
    def set_epsilon(self):
        desc = "Radius of the random uniform noise"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("epsilon_radius", zero_ok=True, target=self.config)
        
    def set_startpoint(self):
        desc = "Starting point of the simulation"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("start_point", target=self.config)
        
    def set_escapedistance(self):
        desc = "Distance at which to consider points as escaped\n(prevents extreme array sizes)"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("escape_distance", target=self.config)
    
    def set_autorun_timesteps(self):
        self.autorun_timesteps = self.config.timestep
        if self._set_attr("autorun_timesteps", zero_ok=True):
            self.autorun_timesteps = max(self.autorun_timesteps-self.config.timestep, 0)
        else: self.autorun_timesteps = 0

        
        
    def _set_attr(self, attr_name, negative_ok=False, zero_ok=False, target=None):
        if target is None: target = self
        current_value = getattr(target, attr_name, None)
        t = type(current_value)
        
        user_input = self._depth_input(f"{current_value} -> ")
        success = False
        if user_input:
            if t in [int, float]:
                user_input = user_input.replace(",",".")
                if RE_INT.match(user_input) or (t is float and RE_FLOAT.match(user_input)):
                    user_input = t(user_input)
                    if user_input>0 or (negative_ok and user_input<0) or zero_ok:
                        setattr(target, attr_name, user_input)
                        success = True
                if not success:
                    complain = "! Given value must be a "
                    if not zero_ok: complain += "non-zero "
                    if not negative_ok: complain += "positive "
                    complain += "integer" if t is int else "float"
                    self._depth_print(complain)
            elif t is str:
                setattr(target, attr_name, user_input)
                success = True
            elif t is UserModifiablePoint:
                m = RE_2D_POINT.match(user_input)
                if m:
                    new_point = UserModifiablePoint(float(m.group(1)), float(m.group(2)))
                    setattr(target, attr_name, new_point)
                    success = True
                else:
                    self._depth_print("Not a recognizable 2 dimensional point")
            else:
                self._depth_print(f"Value setting of type ({t}) is not defined")
        else: self._depth_print("Invalid input")
        if success:
            self._depth_print("Value set")
        print("")
        return success
        
    
    def modifyfunction(self):
        while not self.modifyfunction_options():
            self.function.trim_excess_constants()
        
    def modifyfunction_options(self):
        options = [
            ("Modify the X-axis Equation", self._modifyfunction_axis_x),
            ("Modify the Y-axis Equation", self._modifyfunction_axis_y),
            ("Define Constants", self._modifyfunction_define),
            ("Save Function", self.save_function),
            ("Load Function", self.load_function),
            ("Done", self.quit),
            ]
        desc = f"X-axis: {self.function.x}"
        desc += f"\nY-axis: {self.function.y}"
        
        if self.function.constants:
            desc += "\nDefined:"
            for k,v in self.function.constants.items():
                desc += f"\n  {k} = {v}"
        missing_constants = self.function.missing_constants()
        if missing_constants:
            desc += "\nUndefined: " + ", ".join(missing_constants)
        
        self._show_user_options(desc, options)
        return self._process_user_input(options)

    def _modifyfunction_axis_x(self):
        desc = "Set a new X-axis equation for the function"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        user_input = self._depth_input(self.function.x+" -> ")
        if user_input:
            self.function.x = user_input
            self._depth_print("Value set")
        
    def _modifyfunction_axis_y(self):
        desc = "Set a new Y-axis equation for the function"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        user_input = self._depth_input(self.function.y+" -> ")
        if user_input:
            self.function.y = user_input
            self._depth_print("Value set")

    def _modifyfunction_define(self):
        desc = "Define function constants"
        if self.function.constants:
            desc += "\nDefined:"
            for k,v in self.function.constants.items():
                desc += f"\n  {k} = {v}"
        missing_constants = self.function.missing_constants()
        if missing_constants:
            desc += "\nUndefined: " + ", ".join(missing_constants)
        desc = clean_text_box(desc)
        self._depth_print(desc)

        k = v = None
        
        user_input = self._depth_input("Constant: ")
        if not user_input:
            return
        if user_input in self.function.constants:
            k = user_input
            user_input = self._depth_input(f"Redefine '{user_input}' as: ")
            user_input = user_input.replace(",",".")
            if m:=RE_FLOAT.match(user_input): v = float(user_input)
            elif m:=RE_INT.match(user_input): v = int(user_input)
        elif user_input in missing_constants:
            k = user_input
            user_input = self._depth_input(f"Define '{user_input}' as: ")
            user_input = user_input.replace(",",".")
            if m:=RE_FLOAT.match(user_input): v = float(user_input)
            elif m:=RE_INT.match(user_input): v = int(user_input)
        else:
            self._depth_print("Invalid constant")
        
        if v is not None:
            self.function.constants[k] = v
            self._depth_print("Value set")
        elif k is not None:
            self._depth_print("Invalid value")


    def activerun(self):
        self._activerun_init()
        while not self.activerun_options(): pass
        
    def activerun_options(self):
        options = [
            ("Next Timestep", self._activerun_next),
            ("Open Matplotlib Figure", self._activerun_visuals),
            ("Modify Epsilon", self.set_epsilon),
            ("Modify Function", self.modifyfunction),
            ("Run Until Timestep", self.set_autorun_timesteps),
            ("Run Until Hausdorff", None),
            ("Save State", self.save_timestep),
            ("Load State", self.load_timestep),
            ("More Data...", self._activerun_moredata),
            ("Stop", self.quit),
            ]

##        tl, br = np.divide(self.config.topleft, self.config.precision), np.divide(self.config.bottomright, self.config.precision)
        desc = f"Timestep: {self.config.timestep}"
        desc += f"\nEpsilon radius: {self.config.epsilon_radius} ({int(self.config.epsilon_radius/self.config.precision)} pixels)"
        desc += f"\nFunction: {self.function}"
        if self.autorun_timesteps>0:
            self.autorun_timesteps -= 1
            desc = clean_text_box(desc)
            self._depth_print(desc+"\n")
            self._activerun_next(print_steps=False)
            return False
        
        self._show_user_options(desc, options)
        return self._process_user_input(options)

    def _activerun_moredata(self):
        desc = "Configuration:"
        desc += f"\n Precision: {self.config.precision}"
        desc += f"\n Starting point: {self.config.start_point}"
        desc += f"\n Escape distance: {self.config.escape_distance} ({int(self.config.escape_distance/self.config.precision)} pixels)"
        desc += "\n\nArray:"
        desc += f"\n Domain: ({self.config.topleft[0]} .. {self.config.bottomright[0]}, {self.config.topleft[1]} .. {self.config.bottomright[1]})"
        desc += f"\n Shape: {self.config.image.shape}"
        desc += "\n\nStatistics:"
        desc += f"\n Points processed: {self.stats.processed_points_total} ({self.stats.processed_points} last timestep)"
        desc += f"\n Points escaped: {self.stats.escaped_points_total} ({self.stats.escaped_points} last timestep)"
        desc += f"\n Hausdorff dist: {self.stats.hausdorff_distance} ({self.stats.hausdorff_distance/self.config.precision:.1f} pixels)"
        
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._depth_input("Back...")


    def _activerun_init(self):
        self.config = ModelConfiguration()

    def _activerun_visuals(self):
        self._depth_print("Close matplotlib window to continue", -1)
        self.config.visualization()
        print("")

    def _activerun_fail_check(self): # would it crash
        return bool(self.function.missing_constants())
    
    def _activerun_next(self, print_steps=True):
        if self._activerun_fail_check():
            self._depth_print("Run failed")
            return
        
        if print_steps: self._depth_print("Start")
        #######
        self.config.topleft_old[:] = self.config.topleft

        # create the circle mask & get border points from the image
        eps_circle = self.config.epsilon_circle()
        points = self.config.calc_current_border_points()
        #
        
        # translate indexes to their corresponding points
        self.config.points_as_values(points)

        # process the points using the function
        n = points.shape[0]
        if print_steps: self._depth_print(f"Processing {n} points")
        points[:,0], points[:,1] = self.function(points[:,0], points[:,1])
        self.stats.log_processed_points(n)
        
        # check for escapees and delete them
        escaped_points = np.zeros(points.shape[0], dtype=np.bool_)
        escaped_points |= np.abs(points[:,0]-self.config.start_point.x)>=self.config.escape_distance
        escaped_points |= np.abs(points[:,1]-self.config.start_point.y)>=self.config.escape_distance
        if escaped_points.any():
            points = points[~escaped_points]
            n = escaped_points.sum()
            self.stats.log_escaped_points(n)
            if print_steps: self._depth_print(f"-> {n} of {escaped_points.size} points escaped")
            if points.size==0:
                if print_steps: self._depth_print("-> Aborted")
                return
        #
        
        # measure the new topleft and bottomright
        eps_r = eps_circle.shape[0]/2 # space needed around a point for the epsilon circle
        eps_r *= self.config.precision
        temp_points = np.subtract(points, eps_r)
        self.config.topleft[0] = min(self.config.topleft[0], temp_points[:,0].min())
        self.config.topleft[1] = min(self.config.topleft[1], temp_points[:,1].min())
        temp_points += eps_r*2
        self.config.bottomright[0] = max(self.config.bottomright[0], temp_points[:,0].max())
        self.config.bottomright[1] = max(self.config.bottomright[1], temp_points[:,1].max())


        # translate the points to positive integer format
        self.config.points_as_units(points)
        
        # calculate the new image shape
        image_domain = np.subtract(self.config.bottomright, self.config.topleft)
        image_shape = image_domain/self.config.precision
        image_shape = np.maximum(image_shape.astype(np.int32)+((image_shape%1)!=0), self.config.image_history.shape)

        # create the new image array
        new_image = np.zeros(image_shape, dtype=np.bool_)
        
        # paint the new image with circles
        if print_steps: self._depth_print("Painting")
        eps_r = eps_circle.shape[0]//2
        for x,y in points:
            x = int(x)
            y = int(y)
            x_slice = slice(x-eps_r, x+eps_r+1)
            y_slice = slice(y-eps_r, y+eps_r+1)
            new_image[x_slice, y_slice] |= eps_circle
        
        if self.config.autofill:
            new_image |= fill_closed_areas(new_image)
        #########
        
        if print_steps: self._depth_print("Padding")
        #### PADDING CORRECTION
        # expand the cumulative image array
        pad_l = -(self.config.topleft-self.config.topleft_old) / self.config.precision

        # collect the error from padding misplacement
        # also shift the topleft and bottomright accordingly
        pad_error = pad_l%1
        self.config.padding_error += pad_error
        self.config.topleft += pad_error*self.config.precision
        self.config.bottomright += pad_error*self.config.precision

        # consume full integers of the padding error to adjust the image back to place (towards negative)
        # also shift the topleft and bottomright back in to place
        pad_correction = self.config.padding_error.astype(np.int8)
        if pad_correction.any():
            self.config.padding_error -= pad_correction
            self.config.topleft -= pad_correction*self.config.precision
            self.config.bottomright -= pad_correction*self.config.precision
##            print("PAD_CORRECTION:", pad_correction)
        ####
        
        # do the padding
        shape_diff = np.subtract(image_shape, self.config.image_history.shape)
        pad_r = shape_diff-pad_l.astype(np.int32)
        pad_l += pad_correction
        pad = ((int(pad_l[0]),int(pad_r[0])), (int(pad_l[1]),int(pad_r[1])))
##        print("PAD:", pad)
        
        self.config.image_history = np.pad(self.config.image_history, pad)
        self.config.image = np.pad(self.config.image, pad) # needed for hausdorff
        if pad_correction.any():
            new_image = np.pad(new_image, ((pad_correction[0],0),(pad_correction[1],0)))
        #
        
##        print("same shapes:", new_image.shape, self.config.image.shape, self.config.image_history.shape)
        
        # hausdorff
        if print_steps: self._depth_print("Hausdorff")
        dist = hausdorff_distance(new_image, self.config.image)
        dist = max(dist, hausdorff_distance(self.config.image, new_image))
        dist *= self.config.precision
        self.stats.log_hausdorff(dist)
        if print_steps: self._depth_print(f"-> {dist:.3f}")
        #

        # previous image is no longer needed
        self.config.image = new_image
        #
        
        # add the mask to create a full image
        if self.config.alt_visuals:
            self.config.image_history[self.config.image] = self.config.image[self.config.image]*(self.config.timestep+1)
        else:
            self.config.image_history += self.config.image
        #
        
        self.config.timestep += 1
        if print_steps: self._depth_print("Done")








    

    def list_savefiles(self, ext):
        saves = []
        for f in list_files(SAVE_DIR):
            if f.endswith(f".{ext}"):
                name = os.path.basename(f).rsplit(".", 1)[0]
                size = readable_filesize(getsize(f))
                saves.append((name,size,f))
        
        if saves:
            max_name_len = len(max(saves, key=lambda x:len(x[0])))
            max_size_len = len(max(saves, key=lambda x:len(x[1])))
            
            desc = ""
            for name,size,f in saves:
                if desc: desc += "\n"
                desc += f"{name}"+"."*(10 + max_name_len*2-len(name) + max_size_len*2-len(size))+size
            
            desc = clean_text_box(desc, l=">", pad_t=0, pad_b=0)
            self._depth_print(desc)

    def save_timestep(self, name=None):
        if name is None:
            name = self._depth_input("Save as: ")
        if name:
            obj = (self.config, self.stats, self.function)
            save(os.path.join(SAVE_DIR, name+".timestep"), obj)

    def save_function(self, name=None):
        if name is None:
            name = self._depth_input("Save as: ")
        if name:
            obj = self.function
            save(os.path.join(SAVE_DIR, name+".function"), obj)
    
    def load_timestep(self, name=None):
        if name is None:
            self.list_savefiles("timestep")
            name = self._depth_input("Timestep name: ")
        if name:
            obj = self._load_file(name, "timestep")
            if obj is not None:
                self.config, self.stats, self.function = obj
                
    def load_function(self, name=None):
        if name is None:
            self.list_savefiles("function")
            name = self._depth_input("Function name: ")
        if name:
            obj = self._load_file(name, "function")
            if obj is not None:
                self.function = obj

    def _load_file(self, name, ext):
        if name:
            f = os.path.join(SAVE_DIR, f"{name}.{ext}")
            if os.path.isfile(f):
                obj = load(f)
                if obj is not None:
                    self._depth_print("Success")
                    return obj
                else: self._depth_print("Failed")
            else: self._depth_print(f"File not found: '{f}'")
        else: self._depth_print("Cancelled")

if __name__ == "__main__":
    interface = ConsoleInterface()
    interface.mainmenu()


    # hausdorff testing
##    test_image = circle_mask(3, border=8)
##    test_image2 = np.zeros_like(test_image)
##    test_image2[3,3] = True
##    
##    out = hausdorff_distance(test_image, test_image2)
##    print(out)


    
##    test_image = circle_mask(50, border=5)
##    print(test_image.shape)
##
##    half_image = halve_the_array_size(test_image)
##    print(half_image.shape)
##    quarter_image = halve_the_array_size(half_image)
##    quarter_image = (quarter_image*4).astype(np.int8)
##    print(quarter_image.shape)
##    print(quarter_image)
##
##    print("")
##    third_image = third_the_array_size(test_image)
##    print(third_image.shape)
##    sixth_image = third_the_array_size(third_image)
##    sixth_image = (sixth_image*9).astype(np.int8)
##    print(sixth_image.shape)
##    print(sixth_image)

    









